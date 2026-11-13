# CAD eval set

Sixteen prompts a user might type, to find out where the general CAD agent fails and whether a change helps.
Results are pass/fail counts per case.

## How a case is judged
Each case in `cases.py` has figures fixed by hand from the prompt's own dimensions (overall size, through-holes,
pieces or solid bodies, volume range). The agent's exported STEP is measured with the same code `cad_run` uses and
checked against them with `cad_spec.check`. The agent's own `spec.json` is never trusted; it is only compared with the
case's figures and reported (`spec_gap`).

| category | what it is | cases |
|---|---|---|
| part | stated dimensions | plate, ring, bracket, box, bolt, flange, grid panel, tube, shaft, slotted plate, channel, cup |
| assembly | several solid bodies in one file | box with lid, stacked discs |
| open | loose prompt, loose figures (about 20 %) | phone stand |
| hard | easy to get wrong | spur gear |

## Run it
```bash
scripts/cadeval/eval.sh selftest                       # references pass, top-half-cut copies fail; no model needed
scripts/cadeval/eval.sh run --arm both --repeat 3      # both arms, 3 runs each (about 16 x 2 x 3 agent runs)
scripts/cadeval/eval.sh run --ids part,assembly        # a subset
scripts/cadeval/eval.sh report <outdir>
```
`selftest` needs the CAD Tools Python (Otto: Settings > MCP > Connect CAD Tools once, or `CADEVAL_PYTHON`).
Results go to `../otto-benchcad-runs/cadeval/<stamp>/` (`results.jsonl`, `summary.txt`, trajectories).

## The two arms
`spec` is the general prompt with the spec step (write `spec.json` first; `cad_run` checks against it). `control` is the
shipped `cad-agent` prompt, which has no spec step (`agent_library._CAD_AGENT_PROMPT_WITH_SPEC` against `_CAD_AGENT_PROMPT`).
Both run as fresh agents in the benchmark backend with
identical tools and settings, so only the prompt differs. Neither has the named subagents (drawing-reader,
cad-critic) that the shipped `cad-agent` can call; nothing here needs them.

## Reading the result
- 16 cases, so differences of one or two cases are noise. Use `--repeat` and look at per-case counts, not the total.
- A case that fails in both arms points at a real weakness (or at a case that is wrong: run `selftest`).
- `spec.json written` and `mismatch seen` show whether the agent used the spec step; `spec_gap` shows where its spec
  disagreed with the case's figures.
- Not covered: reading drawings or photos, assemblies that need placement, freeform shapes, and anything over a
  few features. The `phone_stand` figures are too loose to notice a short part (`loose` in `cases.py`).
