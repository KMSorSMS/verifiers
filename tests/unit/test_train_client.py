import asyncio
import base64
import json

import httpx
import numpy as np
import pytest
from openai import AsyncOpenAI
from renderers.base import ParsedResponse, RenderedTokens

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.clients import train
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.errors import InferenceCleanupError, ProviderError
from verifiers.v1.interception.pool import ElasticInterceptionPool
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession


class Renderer:
    supports_tools = True
    is_multimodal = False

    def render(self, messages, **kwargs):
        return RenderedTokens(
            token_ids=[1, 2, 3],
            message_indices=[0, 0, -1],
            sampled_mask=[False, False, False],
            is_content=[False, True, False],
            message_roles=["user"],
        )

    def get_stop_token_ids(self):
        return [99]

    def parse_response(self, completion_ids, **kwargs):
        assert completion_ids == [7, 8]
        return ParsedResponse(content="done")


def make_client(monkeypatch, handler, **options):
    sdk = AsyncOpenAI(
        base_url="http://engine/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(train, "build_async_openai", lambda config: sdk)
    pool = train.ElasticRendererPool("base", None, multiplex=256)
    monkeypatch.setitem(pool._renderers, pool.key, [train.RendererSlot(Renderer())])
    client = train.TrainClient(
        vf.TrainClientConfig(
            backend="sglang", renderer_model_name="base", backend_options=options
        )
    )
    return client, sdk


@pytest.mark.asyncio
async def test_native_request_and_trace_preserve_evidence(monkeypatch):
    requests = []
    array = np.arange(8, dtype=np.int32).reshape(4, 2, 1) + 1000

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        body = json.loads(request.content)
        requests.append((request, body))
        return httpx.Response(
            200,
            json={
                "output_ids": [7, 8],
                "meta_info": {
                    "id": body["rid"],
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "output_token_logprobs": [[-0.1, 7, None], [-0.2, 8, None]],
                    "indexer_topk": base64.b64encode(array.tobytes()).decode(),
                    "weight_version": "step-2",
                    "prefill_weight_versions": [
                        {"version": "step-1", "start": 0, "end": 2},
                        {"version": "step-2", "start": 2, "end": 3},
                    ],
                },
            },
        )

    client, sdk = make_client(
        monkeypatch,
        handler,
        lora_path="adapter",
        return_indexer_topk=True,
        indexer_topk_shape=[2, 1],
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")),
    )
    turn = graph.prepare_turn(trace, [vf.UserMessage(content="hi")])
    try:
        response = await client.get_response(
            ChatDialect(),
            {"model": "base", "messages": [{"role": "user", "content": "hi"}]},
            vf.SamplingConfig(max_tokens=5, seed=19, temperature=0.8),
            session_id="rollout-a",
            turn=turn,
        )
        turn.commit(response)
        restored = vf.Trace.model_validate(trace.model_dump())
        branch = restored.branches[0]
        request, body = requests[0]
        assert request.url.path == "/generate"
        assert body["input_ids"] == [1, 2, 3]
        assert body["sampling_params"]["sampling_seed"] == 19
        assert body["lora_path"] == "adapter"
        assert request.headers["X-Session-ID"] == "rollout-a"
        assert branch.token_ids.tolist() == [1, 2, 3, 7, 8]
        np.testing.assert_allclose(branch.logprobs, [0, 0, 0, -0.1, -0.2])
        assert branch.indexer_topk.dtype == np.int32
        np.testing.assert_array_equal(branch.indexer_topk[:-1], array)
        assert restored.nodes[-1].weight_versions[0].version == "step-2"
        assert restored.nodes[-1].prompt_tokens == 3
        assert restored.nodes[-1].prefill_weight_versions[0].version == "step-1"
        assert restored.nodes[-1].prefill_weight_versions[-1].end == 3
        assert response.raw["choices"][0]["message"]["content"] == "done"
    finally:
        await client.close()
    assert sdk.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,status", [("abort", 503), ("broken", 502)])
async def test_provider_errors_are_not_successful_rollouts(monkeypatch, reason, status):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(
            200, json={"meta_info": {"finish_reason": {"type": reason}}}
        )

    client, _ = make_client(monkeypatch, handler)
    try:
        with pytest.raises(ProviderError) as exc:
            await client.get_response(
                ChatDialect(),
                {"model": "base", "messages": [{"role": "user", "content": "hi"}]},
                vf.SamplingConfig(),
            )
        assert exc.value.status_code == status
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cancellation_aborts_only_the_current_request(monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()
    requests = []

    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/abort_request":
            stopped.set()
            return httpx.Response(200, json={})
        started.set()
        await stopped.wait()
        return httpx.Response(
            200, json={"meta_info": {"finish_reason": {"type": "abort"}}}
        )

    client, _ = make_client(monkeypatch, handler)
    try:
        task = asyncio.create_task(
            client.get_response(
                ChatDialect(),
                {"model": "base", "messages": [{"role": "user", "content": "hi"}]},
                vf.SamplingConfig(),
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert requests[1] == ("/abort_request", {"rid": requests[0][1]["rid"]})
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_type", [InterceptionServer, ElasticInterceptionPool])
@pytest.mark.parametrize("abort_fails", [False, True])
async def test_interception_release_drains_inference_before_acknowledging_cancel(
    monkeypatch, pool_type, abort_fails
):
    """An HTTP handler outlives its caller; releasing its slot must wait for the
    actual GPU abort, and retain a failure even after its trace has been sealed."""
    generating = asyncio.Event()
    aborting = asyncio.Event()
    finish_abort = asyncio.Event()
    generation_stopped = asyncio.Event()

    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        if request.url.path == "/abort_request":
            aborting.set()
            await finish_abort.wait()
            if not abort_fails:
                generation_stopped.set()
            return httpx.Response(503 if abort_fails else 200, json={})
        generating.set()
        await generation_stopped.wait()
        return httpx.Response(
            200, json={"meta_info": {"finish_reason": {"type": "abort"}}}
        )

    client, _ = make_client(monkeypatch, handler, abort_timeout=0.5)
    monkeypatch.setattr(InterceptionServer, "_client", lambda self, config: client)
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="hi")),
    )
    session = RolloutSession(
        ctx=vf.ModelContext(
            model="base", client=vf.TrainClientConfig(backend="sglang")
        ),
        trace=trace,
    )
    calls = []
    try:
        async with pool_type() as pool, httpx.AsyncClient(trust_env=False) as caller:

            async def own_slot():
                async with pool.acquire(session) as (url, secret, _):
                    calls.append(
                        asyncio.create_task(
                            caller.post(
                                f"{url}/v1/chat/completions",
                                headers={"Authorization": f"Bearer {secret}"},
                                json={
                                    "model": "base",
                                    "messages": [{"role": "user", "content": "hi"}],
                                },
                            )
                        )
                    )
                    await asyncio.Event().wait()

            owner = asyncio.create_task(own_slot())
            await asyncio.wait_for(generating.wait(), 3)
            owner.cancel()
            await asyncio.wait_for(aborting.wait(), 3)
            owner.cancel()  # a second cancellation must not orphan abort cleanup
            await asyncio.sleep(0)
            assert session.released
            assert not owner.done()
            finish_abort.set()
            with pytest.raises(
                InferenceCleanupError if abort_fails else asyncio.CancelledError
            ):
                await asyncio.wait_for(owner, 3)
            assert bool(session.cleanup_error) is abort_fails
            assert not session.tasks
            assert not trace.calls  # sealed before the late provider failure
            if abort_fails:
                with pytest.raises(InferenceCleanupError):
                    await session.refused()
            await asyncio.gather(*calls, return_exceptions=True)
    finally:
        finish_abort.set()
        for task in calls:
            task.cancel()
        await asyncio.gather(*calls, return_exceptions=True)
        await client.close()
