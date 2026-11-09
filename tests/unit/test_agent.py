"""`Interaction` segments: a failed segment reports its own partial reply, and an
expired agent deadline stops the rollout as a timeout."""

import asyncio
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction, _EpisodeAgent
from verifiers.v1.configs.runtime import SubprocessConfig
from verifiers.v1.env import Env
from verifiers.v1.errors import InferenceCleanupError
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.null.harness import NullHarness, NullHarnessConfig
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.session import RolloutLimits
from verifiers.v1.types import AssistantMessage


class FailingSegmentRollout:
    ok = Rollout.ok
    closed = Rollout.closed
    fail = Rollout.fail
    step = Rollout.step
    timeout = Rollout.timeout


def make_trace() -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=0, prompt=None),
            key="dataset/example-0",
            hash="content-digest",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["abort", "close"])
async def test_rollout_cleanup_failure_survives_teardown_and_is_not_retryable(method):
    run = Rollout(
        task=vf.Task(vf.TaskData(prompt="hi")),
        agent_config=vf.AgentConfig(),
        harness=NullHarness(NullHarnessConfig()),
        ctx=vf.ModelContext(model="base", client=vf.TrainClientConfig()),
        runtime_config=SubprocessConfig(),
        timeouts=RolloutTimeouts(),
        limits=RolloutLimits(),
    )
    closed = []
    failure = InferenceCleanupError("engine abort failed")

    async def release_interception():
        run._session.release()
        run._session.cleanup_error = failure
        raise failure

    async def close_deployment():
        closed.append("deployment")

    run._stack.push_async_callback(release_interception)
    run.deployment = SimpleNamespace(close=close_deployment)
    with pytest.raises(InferenceCleanupError, match="engine abort failed"):
        await getattr(run, method)()
    assert closed == ["deployment"]
    assert run.trace.last_error.type == "InferenceCleanupError"
    assert not run.trace.ok
    with pytest.raises(InferenceCleanupError):
        await run.close()  # an idempotent close cannot report false success


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_path", ["group", "cancel", "finalize"])
async def test_episode_propagates_inference_cleanup_failure(failure_path):
    failure = InferenceCleanupError("unconfirmed engine abort")

    class CleanupEnv:
        config = vf.EnvConfig()

        def _episode_agents(self, ctx, traces, *callbacks):
            traces.append(make_trace())

        async def setup(self, agents):
            pass

        async def run(self, task, agents):
            if failure_path == "group":

                async def failed_agent():
                    raise failure

                async with asyncio.TaskGroup() as group:
                    group.create_task(failed_agent())
            elif failure_path == "cancel":
                raise asyncio.CancelledError() from failure

        async def finalize(self, task, episode):
            raise failure

    with pytest.raises(InferenceCleanupError, match="unconfirmed engine abort"):
        await Env.run_episode(CleanupEnv(), vf.Task(vf.TaskData(prompt="hi")), None)


def make_rollout(trace: vf.Trace, session, time_remaining=None):
    run = FailingSegmentRollout()
    run.trace = trace
    run._opened = True
    run._closed = False
    run._failed = False
    run._failure = None
    run._borrowed_runtime = None
    run.runtime = None
    run._agent_time_remaining = time_remaining
    run._timeouts = RolloutTimeouts()
    run._harness_session = session
    run.deployment = SimpleNamespace(run=lambda turn: turn)
    run._session = SimpleNamespace(
        request_interceptors=[],
        error=None,
        stopped=False,
    )
    run.deadline_at = None
    return run


@pytest.mark.asyncio
async def test_failed_segment_does_not_reuse_prior_root_reply():
    trace = make_trace()
    trace.root_reply = "previous reply"

    class FailingSession:
        async def turn(self, messages):
            trace.nodes.append(
                MessageNode(
                    parent=None,
                    message=AssistantMessage(content="current partial reply"),
                    sampled=True,
                )
            )
            raise RuntimeError("segment failed after sampling")

    run = make_rollout(trace, FailingSession())

    segment = await Interaction(run).turn("next")

    assert segment.last_reply == "current partial reply"
    assert trace.root_reply is None
    assert trace.last_reply == "current partial reply"


@pytest.mark.asyncio
async def test_expired_deadline_is_a_timeout_when_the_runtime_reports_an_error():
    """A runtime can turn the deadline's cancellation into its own error, as Prime's SDK
    does for a cancelled request. The rollout still stops on its agent timeout."""
    trace = make_trace()

    class CancelReportingSession:
        async def turn(self, messages):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise RuntimeError("request was cancelled") from None

    run = make_rollout(trace, CancelReportingSession(), time_remaining=0.05)

    await Interaction(run).turn("next")

    assert trace.is_timeout and not trace.errors
    assert trace.stop_condition == "agent_timeout"


@pytest.mark.asyncio
async def test_episode_runs_have_independent_sampling_without_mutating_the_agent(
    monkeypatch,
):
    """A repeated role gets distinct seeds before the concurrency gate; the actual
    rollout and recorded agent identity agree, including explicit overrides."""
    contexts = []

    class FinishedRollout:
        ok = True

        def __init__(self, *, ctx, agent_config, on_trace, trace_id, **kwargs):
            self.trace = make_trace()
            self.trace.id = trace_id
            self.trace.agent.config = agent_config
            contexts.append(ctx)
            if on_trace is not None:
                on_trace(self.trace)

        async def open(self):
            await asyncio.sleep(0)
            return True

        async def step(self):
            pass

        async def close(self):
            self.trace.ok = True
            return self.trace

    monkeypatch.setattr("verifiers.v1.agent.Rollout", FinishedRollout)
    completed = []
    agent = _EpisodeAgent(
        vf.AgentConfig(
            model="base-model",
            runtime=SubprocessConfig(),
            harness=NullHarnessConfig(id="null"),
            sampling=vf.SamplingConfig(seed=40, temperature=0.7, top_p=0.9),
            max_turn_tokens=16,
        ),
        interception=None,
        name="solver",
        shared_tools={},
        task_cls=vf.Task,
        gate=asyncio.Semaphore(1),
        completed=completed,
        on_trace=None,
        on_discard=None,
        warned_resources=set(),
    )
    task = vf.Task(vf.TaskData(idx=0, prompt="question"))
    traces = await asyncio.gather(
        agent.run(task),
        agent.run(task),
        agent.run(task, sampling=vf.SamplingConfig(seed=91, max_tokens=100)),
        agent.run(task, sampling=vf.SamplingConfig(seed=None)),
    )
    assert [ctx.sampling.seed for ctx in contexts] == [40, 41, 91, None]
    assert [trace.agent.config.sampling.seed for trace in traces] == [40, 41, 91, None]
    assert all(
        ctx.sampling.temperature == 0.7 and ctx.sampling.top_p == 0.9
        for ctx in contexts
    )
    assert all(ctx.sampling.max_tokens == 16 for ctx in contexts)
    assert agent.ctx.sampling.seed == 40
    assert agent.config.sampling.seed == 40
    assert len(completed) == 4
