"""Measure aggregate decode/prefill throughput at N concurrent requests against an
OpenAI-compatible local server (oMLX, exo). BENCH_URL / BENCH_MODEL override the target."""
import asyncio, os, time, httpx, random
URL=os.environ.get("BENCH_URL","http://127.0.0.1:52414/v1/chat/completions"); M=os.environ.get("BENCH_MODEL","mlx-community--Qwen3.6-35B-A3B-8bit")
def prompt(kind, i):
    if kind=="decode": return f"[{i}] Write a long numbered list of distinct fruit facts. /no_think"
    words=" ".join(random.choice(["alpha","bravo","delta","omega","sigma","kappa","zeta"])+str(random.randint(0,9999)) for _ in range(2500))
    return f"[{i}] Summarize in one word: {words} /no_think"
async def one(c,kind,i):
    t=time.time(); n=0
    r=await c.post(URL,json={"model":M,"messages":[{"role":"user","content":prompt(kind,i)}],"max_tokens":200 if kind=="decode" else 8,"temperature":0})
    r.raise_for_status(); u=r.json()["usage"]; return time.time()-t,u["completion_tokens"],u["prompt_tokens"]
async def run(kind,n):
    async with httpx.AsyncClient(timeout=600) as c:
        t=time.time(); res=await asyncio.gather(*(one(c,kind,i+random.randint(0,10**6)) for i in range(n))); w=time.time()-t
    toks=sum(r[1] for r in res); pt=sum(r[2] for r in res)
    print(f"{kind:7} N={n} wall={w:6.1f}s  out_tok={toks:5d} agg_decode={toks/w:6.1f} tok/s  prompt_tok={pt} agg_prefill={pt/w:7.0f} tok/s  per-req mean={sum(r[0] for r in res)/n:5.1f}s",flush=True)
async def main():
    await run("decode",1)  # warm/load
    for kind in ("decode","prefill"):
        for n in (1,2,4,8): await run(kind,n)
asyncio.run(main())
