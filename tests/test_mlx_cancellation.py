"""Stopping a session must stop the MLX generation it was waiting on.

The Stop button cancels the asyncio task running the agent.  That
cancellation does reach the model call (LangGraph node -> ``ainvoke`` ->
``_agenerate_with_cache`` -> ``_agenerate``), but a worker thread cannot be
cancelled: while ``_agenerate`` simply awaited ``asyncio.to_thread(...)``, the
thread decoded on to ``max_tokens`` holding the process-wide ``MLX_GEN_LOCK``,
and every other MLX request waited behind a generation nobody would read.

No model is loaded here.  Chat models are built with pydantic's
``model_construct`` (skips ``__init__`` and its weight loading) and the lazily
imported ``stream_generate`` is replaced by :class:`FakeTokenStream`.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

mlx_lm = pytest.importorskip("mlx_lm")
mlx_vlm = pytest.importorskip("mlx_vlm")

import mlx.core as mx
from langgraph.graph import END, START, MessagesState, StateGraph

from chat_models.mlx._shared import MLX_GEN_LOCK
from chat_models.mlx.chat_mlx_text import ChatMLXText
from chat_models.mlx.chat_vlm import MLXVLChatModel
from chat_models.mlx_turbo import _executor as turbo_executor
from chat_models.mlx_turbo.chat import TurboMLXChat

# Tokens decoded before the test presses "Stop".
TOKENS_BEFORE_STOP = 3
# How long the worker may take to hand MLX_GEN_LOCK back after the cancel:
# generous for a loaded machine, yet well under the ~5 s an unstopped
# FakeTokenStream keeps holding it.
LOCK_RELEASE_TIMEOUT_S = 2.0


class FakeTokenStream:
    """Stand-in for ``stream_generate``: one ``"x"`` token every *delay* seconds."""

    def __init__(self, total: int = 500, delay: float = 0.01) -> None:
        self.total = total
        self.delay = delay
        self.produced = 0
        self.closed_holding_lock = None
        self._abort = threading.Event()

    def __call__(self, *args, **kwargs):
        return self._tokens()

    def _tokens(self):
        try:
            for n in range(1, self.total + 1):
                if self._abort.is_set():
                    return
                time.sleep(self.delay)
                self.produced = n
                yield SimpleNamespace(
                    text="x", prompt_tokens=5, prompt_tps=100.0,
                    generation_tokens=n, generation_tps=10.0, peak_memory=0.0,
                )
        finally:
            # The real stream_generate synchronises the GPU when it is closed,
            # so it must be closed while MLX_GEN_LOCK is still held.
            self.closed_holding_lock = MLX_GEN_LOCK.locked()

    def abort(self) -> None:
        """Test cleanup: end a stream that nothing else stopped."""
        self._abort.set()


@pytest.fixture
def token_stream(monkeypatch):
    stream = FakeTokenStream()
    monkeypatch.setattr(mlx_lm, "stream_generate", stream)
    monkeypatch.setattr(mlx_vlm, "stream_generate", stream)
    yield stream
    stream.abort()
    turbo_executor.shutdown(wait=True)  # no-op unless a TurboMLXChat test started it
    assert MLX_GEN_LOCK.acquire(timeout=10), "fake generation never released MLX_GEN_LOCK"
    MLX_GEN_LOCK.release()


def _text_model(monkeypatch, cls=ChatMLXText):
    # The real sampler kwargs build mlx samplers and consume the process-wide
    # loop-recovery temperature bump; neither matters to a fake token stream.
    monkeypatch.setattr(ChatMLXText, "_sampler_kwargs", lambda self: {})
    llm = cls.model_construct(model_path="fake/mlx-model")
    llm._model = object()
    llm._tokenizer = object()  # no chat template -> plain-text prompt
    return llm


def _turbo_model(monkeypatch):
    return _text_model(monkeypatch, cls=TurboMLXChat)


def _vlm_model(monkeypatch):
    monkeypatch.setattr(MLXVLChatModel, "_to_prompt", lambda self, messages, images: "prompt")
    llm = MLXVLChatModel.model_construct(model_path="fake/mlx-vlm")
    llm._model = object()
    llm._processor = object()
    llm._config = object()
    return llm


MODEL_BUILDERS = pytest.mark.parametrize(
    "build", [_text_model, _turbo_model, _vlm_model],
    ids=["ChatMLXText", "TurboMLXChat", "MLXVLChatModel"],
)


async def _stop_mid_generation(run, stream):
    """Cancel the coroutine *run* a few tokens in, as the Stop button does.

    Returns ``(lock_released, tokens_decoded_after_the_cancel)``.
    """
    task = asyncio.create_task(run)
    deadline = time.monotonic() + 10
    while stream.produced < TOKENS_BEFORE_STOP:
        if task.done():
            task.result()  # surface why the generation ended early
            pytest.fail("generation finished before it could be stopped")
        assert time.monotonic() < deadline, "fake generation never started"
        await asyncio.sleep(0.005)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    produced_at_cancel = stream.produced

    released = MLX_GEN_LOCK.acquire(timeout=LOCK_RELEASE_TIMEOUT_S)
    if released:
        MLX_GEN_LOCK.release()
    return released, stream.produced - produced_at_cancel


@MODEL_BUILDERS
async def test_cancelling_agenerate_stops_the_worker_and_frees_the_lock(
    build, monkeypatch, token_stream,
):
    llm = build(monkeypatch)

    released, extra_tokens = await _stop_mid_generation(
        llm._agenerate([HumanMessage("hi")]), token_stream,
    )

    assert released, (
        f"MLX_GEN_LOCK still held {LOCK_RELEASE_TIMEOUT_S}s after the cancel; "
        f"the orphaned worker decoded {extra_tokens} more tokens meanwhile"
    )
    # At most the token that was already being decoded when Stop arrived.
    assert extra_tokens <= 1
    # The stream was closed (GPU synchronised) before the lock was handed back.
    assert token_stream.closed_holding_lock is True


async def test_stopping_a_langgraph_run_stops_the_mlx_generation(monkeypatch, token_stream):
    """The production path: Stop cancels the task draining ``graph.astream``."""
    llm = _text_model(monkeypatch)

    async def call_model(state):
        return {"messages": [await llm.ainvoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    graph = builder.compile()

    async def drain():  # like backend.routes.sessions._run_agent_stream_loop
        async for _ in graph.astream(
            {"messages": [HumanMessage("hi")]}, stream_mode=["values", "messages"],
        ):
            pass

    released, extra_tokens = await _stop_mid_generation(drain(), token_stream)

    assert released, f"MLX_GEN_LOCK still held; {extra_tokens} tokens decoded after Stop"
    assert extra_tokens <= 1


async def test_generation_after_a_cancelled_one_is_unaffected(monkeypatch, token_stream):
    llm = _text_model(monkeypatch)
    budget_checks, cache_clears = [], []
    monkeypatch.setattr(ChatMLXText, "_enforce_cache_budget", lambda self: budget_checks.append(1))
    monkeypatch.setattr(mx, "clear_cache", lambda: cache_clears.append(1))

    released, _ = await _stop_mid_generation(llm._agenerate([HumanMessage("hi")]), token_stream)
    assert released

    # The cancelled call still runs the usual post-generation cleanup, on its
    # worker thread, right after releasing the lock.
    deadline = time.monotonic() + LOCK_RELEASE_TIMEOUT_S
    while not cache_clears and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert budget_checks and cache_clears

    # The stop request belonged to that call only: the next one runs to the end.
    follow_up = FakeTokenStream(total=4, delay=0)
    monkeypatch.setattr(mlx_lm, "stream_generate", follow_up)
    result = await llm._agenerate([HumanMessage("again")])
    assert result.generations[0].message.content == "xxxx"
    assert follow_up.produced == 4


@MODEL_BUILDERS
def test_sync_invoke_without_a_cancel_event_runs_to_completion(build, monkeypatch, token_stream):
    token_stream.total, token_stream.delay = 20, 0
    llm = build(monkeypatch)

    message = llm.invoke([HumanMessage("hi")])

    assert token_stream.produced == 20
    assert message.content == "x" * 20
