# CAD agent

Otto can build 3D CAD from a description, a drawing, a photo, or a STEP file. The `cad-agent` writes a
[CadQuery](https://cadquery.readthedocs.io/) program, runs it in a sandbox, renders it, and exports STEP / STL /
OBJ (or a Blender `.blend`). It replaces the old live-Blender integration.

## Try it

1. **Agents page, Tools tab:** find **CAD Tools** and click **Connect** (or **Start**). It ships disabled. The first
   time it installs CadQuery in its own Python 3.12 environment, which takes a minute or two. No Docker needed.
2. Start a session with **cad-agent**. Map a project folder to the session if you can: files are written there
   (otherwise in the session's own files folder).
3. Ask for a part, for example:

   > Make a round flange: 80 mm outside diameter, 8 mm thick, a 30 mm centre bore, and six 8 mm holes evenly spaced
   > on a 60 mm bolt circle. Show me a render, and export STEP and STL.

The agent writes `part.py`, runs and renders it, and exports to `exports/` in the work folder. Renders are PNGs in
`.cad/render/`. On a Mac, select the `.stl` in Finder and press Space to preview it; a `.step` needs a CAD program.
Check the result against what you asked for (for the flange: seven holes, six on the bolt circle plus the bore).

## What is in the box

| Piece | What it is |
|---|---|
| `cad-tools` MCP | [`backend/builtin_mcps/cad_tools/`](../backend/builtin_mcps/cad_tools): `cad_run`, `cad_render`, `cad_export`, `cad_compare`, `step_inspect`, `answer_checkpoint`, plus tools for the benchmark (`cad_autoplace`, `assembly_status`, `budget_status`, `netlist_validate`). Programs run under Seatbelt on macOS (no network, writes only to the work folder). |
| `cad-agent` | The general agent. Subagents: `drawing-reader` (dense drawing sheets) and `cad-critic` (second opinion on a render). |
| `benchcad-agent` | Headless benchmark agent for BenchCAD; not for interactive use. |
| Folder confinement | [`backend/cad_workdir_middleware.py`](../backend/cad_workdir_middleware.py): CAD tools only touch the session's mapped folder. |

`benchcad-agent`, `drawing-reader`, `cad-critic` and `swebench-coding-agent` are installed but hidden from the Agents
list: scripts or `cad-agent` start them by name.

## Checks the agent gets

- `cad_run` reports size, volume, solids, faces, and topology (pieces, through-holes, sealed voids) after every run.
- For a 2x2 sheet of four views of one part, `cad_compare` scores outline and interior edges against the sheet and
  tracks the best version. With part pictures it also gives a shape *gain*: how far the part beats a plain box or
  cylinder at the same scale.
- Optional `spec.json` in the work folder (see [`cad_spec.py`](../backend/builtin_mcps/cad_tools/cad_spec.py)): when
  present, `cad_run` and `cad_export` check the result against stated size, through-holes, pieces and volume and print
  `SPEC OK` or `SPEC MISMATCH`. The shipped `cad-agent` prompt does not ask for it (see below).

## What is known and not known

- **Plan-first was tried and taken out.** Asking the agent to write `spec.json` before any code did not help in a first
  A/B run on a local model (7 of 10 right with it, 8 of 9 without, run stopped at 19 of 32): it wrote wrong specs
  (forgot a bore, called a tube a solid bar) that its own check then confirmed. The prompt with the step is kept as
  `_CAD_AGENT_PROMPT_WITH_SPEC` for the eval.
- **BenchCAD scores are low and noisy.** On `task4/case1` (a chuck assembly from four views) the local Qwen3.6 35B
  scored between 0.000 and 0.015 across versions, one run each. Dev-harness scores are not comparable with reported
  BenchCAD numbers. See [`scripts/benchcad/README.md`](../scripts/benchcad/README.md).
- **The eval set is small.** [`scripts/cadeval/`](../scripts/cadeval) has 16 text prompts with fixed expected figures,
  scored by measuring the exported STEP. It shows large effects only, and covers no drawings, photos, or assemblies that
  need placement.
- **Right numbers, wrong shape passes.** Size, hole and volume checks do not see a part that looks wrong.
- **Not tested here:** a real-user session in the Otto app end to end, and models other than the local Qwen.

## Run the checks

```bash
scripts/cadeval/eval.sh selftest        # reference programs pass their own check; no model needed
.venv/bin/python -m pytest tests/test_cad_tools.py tests/test_cad_autoplace.py tests/test_cad_spec.py tests/test_cadeval.py
```
