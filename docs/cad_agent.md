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
| `cad-tools` MCP | [`backend/builtin_mcps/cad_tools/`](../backend/builtin_mcps/cad_tools): `cad_run`, `cad_render`, `cad_export`, `cad_compare`, `step_inspect`, `answer_checkpoint`, and a few helpers for multi-part work. Programs run under Seatbelt on macOS (no network, writes only to the work folder). |
| `cad-agent` | The general agent. Subagents: `drawing-reader` (dense drawing sheets) and `cad-critic` (second opinion on a render). |
| Folder confinement | [`backend/cad_workdir_middleware.py`](../backend/cad_workdir_middleware.py): CAD tools only touch the session's mapped folder. |

`drawing-reader` and `cad-critic` are installed but hidden from the Agents list: `cad-agent` calls them by name.

## Checks the agent gets

- `cad_run` reports size, volume, solids, faces, and topology (pieces, through-holes, sealed voids) after every run.
- For a 2x2 sheet of four views of one part, `cad_compare` scores outline and interior edges against the sheet and
  tracks the best version. With part pictures it also gives a shape *gain*: how far the part beats a plain box or
  cylinder at the same scale.
- Optional `spec.json` in the work folder (see [`cad_spec.py`](../backend/builtin_mcps/cad_tools/cad_spec.py)): when
  present, `cad_run` and `cad_export` check the result against stated size, through-holes, pieces and volume and print
  `SPEC OK` or `SPEC MISMATCH`. The shipped `cad-agent` prompt does not ask for one; add it yourself if you want the checks.

## Limits

- A part with the right size, hole count and volume passes those checks even if it looks wrong. Look at the render.
- Quality depends on the model. A small local model can miss a feature (a centre bore, a slot) or get proportions wrong:
  check hole counts and overall size against what you asked for.
- Not yet tried end to end in the app by a user: enable CAD Tools, chat with `cad-agent`, view a render.

## Run the tests

```bash
.venv/bin/python -m pytest tests/test_cad_tools.py tests/test_cad_autoplace.py tests/test_cad_spec.py
```
