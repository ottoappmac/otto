# Plan: concurrent LLM requests for Otto on mlx / oMLX / exo

Status: research + plan only. No product code changed. One benchmark script added
(`scripts/llm_concurrency_bench.py`).

## 1. Where the sequencing actually comes from

| Layer | What happens today | Evidence |
|---|---|---|
| In-process `mlx` (`ChatMLXText`, VLM) | **Hard serial.** One process-wide `MLX_GEN_LOCK` around every `stream_generate`; added to avoid Metal command-buffer aborts. | `src/chat_models/mlx/_shared.py:83` |
| `mlx_turbo` | Single-worker executor, serial by design. | `src/chat_models/mlx_turbo/_executor.py` |
| oMLX | Client is a plain `ChatOpenAI`, so concurrent calls reach the server. Server batches (`max_concurrent_requests: 8` in this machine's `~/.omlx/settings.json`). | `src/deep_agent/model_factory.py:748`, measured below |
| exo | Same `ChatOpenAI` client. Server-side batching is reported on by default (`--no-batch` disables). | `model_factory.py:915`; upstream PR text (see 5) |
| Subagent fan-out | `SubagentConcurrencyMiddleware` caps `task` calls at **2** on local providers, per session, by count. | `src/middleware/subagent_concurrency.py`, `Environment.get_max_parallel_subagents` |
| Pre-turn side calls | Run-depth chooser is awaited before the turn (`session_manager.py:5448`); memory relevance ranking runs inside `awrap_model_call` (`backend/memory_relevance.py:530`). I did not confirm whether they overlap each other, and I assumed they are serial. | code read only |
| Memory extraction / ambient / eval | Separate `ainvoke`s from background tasks, uncoordinated with foreground. | grep only |

Takeaway: for oMLX/exo the transport is already concurrent. The sequential feel comes
from (a) the cap of 2, (b) serial pre-turn calls, (c) agent structure (orchestrator waits
on subagents), and (d) for in-process mlx, the lock.

## 2. What the hardware gives you (measured)

`scripts/llm_concurrency_bench.py`, oMLX 0.6.3, Qwen3-8B-4bit, M-series Mac with 128 GB,
one run per cell, distinct prompts (no prefix-cache hits):

| N concurrent | Decode-heavy (200 out tok) aggregate | Prefill-heavy (~13k prompt tok each) aggregate |
|---|---|---|
| 1 | 20 tok/s | 501 tok/s |
| 2 | 31 tok/s (1.6x) | 629 tok/s |
| 4 | 57 tok/s (2.8x) | 560 tok/s |
| 8 | 81 tok/s (4.0x) | 732 tok/s |

Reading: **decode scales** (memory-bandwidth bound, batching is nearly free), **prefill
does not** (compute bound, roughly 500-700 tok/s regardless of N, so per-request latency
at N=8 was 102 s vs 26 s). Wall time for decode N=8 was 19.8 s vs 10 s for N=1, i.e. 8
requests for about 2x the time.

Consequence: concurrency helps fan-outs whose cost is generation, and does little for
fan-outs of long-context subagents (the research case that motivated the cap, 50-70k
tokens each). For those the win is prefix/KV reuse, not parallelism.

## 3. Plan, in priority order

**P0. Measure in Otto, not just against the server.** Add a trace (start/end per LLM call,
session, role: orchestrator / subagent / relevance / extraction / depth) so we can see
the real timeline of one multi-subagent turn. Everything below is a hypothesis until this
shows where idle gaps are. Acceptance: a timeline for a 4-subagent turn on oMLX.

**P1. Replace the count cap with a budgeted, backend-aware gate.**
- One process-wide gate per backend (oMLX URL, exo URL), not per session, so ambient,
  memory and extra sessions count against the same budget.
- Admit by estimated KV/context tokens plus a request count ceiling, not "2". Short
  decode-heavy subagents can then run 4-8 wide; 60k-token ones stay at 1-2.
- Ceiling comes from the server: read oMLX `scheduler.max_concurrent_requests` via the
  admin API Otto already uses; for exo use a conservative default and an override.
- Keep `max_parallel_subagents` as the manual override; `auto` becomes the new logic.

**P1. Overlap the pre-turn side calls.** Run depth selection and memory ranking with
`asyncio.gather` where they do not depend on each other. Small, low-risk gain (they are
short prompts, so prefill contention is minor). Confirm first that they really are serial.

**P2. Two lanes: foreground vs background.** Interactive turns get priority; memory
extraction, ambient and eval wait or run at lower priority. Client-side first (two
semaphores); oMLX also exposes `prefill_priority` and `decode_fairness`, worth testing.

**P2. Make the orchestrator fan out when it should.** Check prompts in
`agent_library.py` encourage independent `task` calls in one message; deepagents then runs
them concurrently. Only valuable once P1 lets them run.

**P3. In-process `mlx`: do not parallelize.** Lifting `MLX_GEN_LOCK` risks the Metal abort
it was added for. Real batching would mean embedding `mlx_lm`'s `BatchGenerator` (what
oMLX does). Recommendation: steer users who want parallelism to oMLX; document that
`mlx` is serial.

**P3. exo.** Benchmark the same script against a real cluster, single node and
pipeline/tensor placements, before committing to defaults. Do not assume oMLX numbers
transfer.

## 4. Risks
- Memory: each concurrent long request holds a KV cache; the original cap exists because
  of "Prefill context too large for available memory". Budget by tokens, keep the override.
- Prefill contention can make a short foreground request wait behind a background prefill
  (the N=8 latency above). The lane split in P2 addresses this.
- Reasoning-model ReAct wrappers (`MLXReActWrapper`) and loop-recovery temperature bumps
  are process-wide state (`consume_temperature_bump`); under real concurrency a bump
  could land on the wrong request. Needs a look before raising parallelism.

## 5. Sources
- oMLX README (continuous batching via mlx-lm `BatchGenerator`, `--max-concurrent-requests` default 8, memory guard): https://github.com/jundot/omlx
- exo README (tensor parallel 1.8x on 2 devices, 3.2x on 4; silent on request concurrency): https://github.com/exo-explore/exo
- exo PR #2312, interleaved prefill with active decode, **open, not merged**, single-node and tensor-parallel text only; pipeline-parallel stays on the synchronous path: https://github.com/exo-explore/exo/pull/2312
- vllm-mlx docs (third-party, same `BatchGenerator` design, vendor-reported speedups): https://vllm-mlx.is-a.dev/guides/continuous-batching/

## 6. Self-evaluation (what I am and am not sure of)

Solid:
- The `MLX_GEN_LOCK`, turbo executor, and subagent cap are read straight from the code.
- The oMLX scaling numbers are real measurements from this machine.

Weak or unverified:
- **One benchmark run per cell, one model, one machine, while the oMLX server may have had
  other load.** The first N=1 decode run (15 tok/s) vs the second (20 tok/s) shows
  roughly 30% noise, so treat the 1.6x/2.8x/4.0x as approximate. The prefill row is
  also noisy, and my prompts were about 13k tokens, not the 50-70k that matter most.
- **I did not trace Otto's real call timeline.** The claim that the cap of 2, serial
  pre-turn calls, and agent structure explain the sequential feel is inferred from code.
  I could not rule out a different dominant cause (e.g. the orchestrator rarely emitting
  parallel `task` calls, or time spent in tools rather than the LLM). P0 exists for this.
- **Pre-turn serialization is assumed.** I saw the call sites but did not confirm ordering.
- **exo is the weakest part.** README is silent on concurrency; "batching on by default"
  comes from a search snippet describing a PR's `--no-batch` help text; PR #2312 is not
  merged, so I cannot say what exo main does under concurrent requests, and I ran no exo
  test. Treat every exo statement as unconfirmed.
- **mlx-lm's own server:** I could not confirm its batching behavior (docs only say
  quantized KV disables batching), so it is excluded from the plan.
- Token-budget admission is a design sketch; I have not checked that Otto can estimate
  per-subagent context size at admit time.
- The benchmark loaded `Qwen3-8B-4bit` into the user's running oMLX server as a side
  effect; the server's idle-eviction setting determines whether it stays resident.
- The gains (P1) are unquantified for Otto end to end. Expect big wins only for
  decode-heavy fan-outs.

## 7. Update: evidence from three real sessions (`~/Library/Application Support/Otto/sessions`)

Computed from the per-call `stats` in `*.messages.json` (LLM time only; messages carry no
timestamps, so tool/wall time is unknown).

| Session | LLM calls | Prefill time | Decode time | Tokens from prefix cache vs prefilled |
|---|---|---|---|---|
| ce82e7a1 | 53 | 83 s | 404 s | 3.9M vs 0.28M |
| 8121df2f | 15 | 2 s | 206 s | 1.1M vs 0.03M |
| 13900093 | 18 | 48 s | 215 s | 1.2M vs 0.16M |

- LLM time is **80-99% decode**, single stream at roughly 70-85 tok/s.
- Prefix caching already removes ~90%+ of prefill.
- Only one subagent ever ran (`coding-agent #1`); there was **no subagent fan-out**. The
  cap of 2 was never the limit in these sessions.
- The LLM calls form a dependent chain (each step needs the previous result), so request
  concurrency cannot overlap them. Some steps issued 2-5 tool calls at once, but those
  are tools, not LLM requests.

Revised view: for these workloads the main lever is **faster single-stream decode**
(speculative decoding with a draft model, shorter outputs, fewer steps), not concurrent
requests. Concurrency still pays for real fan-outs, several sessions at once, and the
small pre-turn/background calls. Caveat: three sessions, one user, one model (unknown to
me which), and I did not measure wall-clock waiting on tools.
