# CAD Tools MCP

Tools for agents that write CadQuery programs. Used by the `cad-agent` (general CAD) and `benchcad-agent` (benchmark).

| Tool | Does |
|---|---|
| `cad_run` | Run a program; report errors, size, volume, solids/faces/edges |
| `cad_render` | Run it and render views (default: four tetrahedral cameras, 2x2) |
| `cad_compare` | Score a part against a 2x2 four-view picture (outline, interior edges, shape gain over a plain block) |
| `cad_autoplace`, `assembly_status` | BenchCAD assembly helpers: place parts from instance pictures; checklist of what is left |
| `cad_export` | Write `step` (universal CAD), `stl`, `obj`, or a Blender `.blend` (needs Blender) |
| `step_inspect` | Bounding box, volume, cylinder radii and axes of a STEP file |
| `answer_checkpoint` | Save the last version that ran cleanly, or restore it |
| `budget_status` | Read the BenchCAD token meter (`METER_URL`) |
| `netlist_validate` | Check a PCB graph the way BenchCAD's scorer does |

**Design spec** (`spec.json` in the work folder), optional and not asked for by the shipped `cad-agent` prompt: a
first A/B run found that asking the model to write it first did not help, because it wrote wrong specs (a missed
bore, a tube as a solid bar) that its own check then confirmed. When a spec.json exists, `cad_run` and `cad_export` check each result against it and print `SPEC OK` or `SPEC MISMATCH` with the
difference in numbers: overall size per axis (`null` = unspecified, `size_tol` default 5 %, `axes: "any"` to ignore
order), through-holes, pieces, sealed voids, volume range. Only what the spec states is checked; no spec, no check.
The format is documented in `cad_spec.py`. It catches wrong proportions and missing or invented holes for any
design, with no reference picture; it cannot tell a wrong shape that has the right size and counts.

CadQuery is installed in this MCP's own Python 3.12 venv when you enable it (same pins as the BenchCAD
sandbox). No Docker needed.

**Where programs run** (`CAD_EXEC`): `auto` (default) uses Seatbelt on macOS: no network, writes only to the work
folder and temp, `~/.ssh` / `~/.aws` / keychains / Otto's data unreadable. `docker` uses the BenchCAD image.
`local` has no isolation.

**Which folder**: the session's mapped project folder, or its own files folder when none is mapped. The agent sends
`workdir="."`; anything outside is refused (`backend/cad_workdir_middleware.py`).

The server process itself is not Seatbelt-wrapped (it writes into task folders and starts the sandboxed child);
see `UNSANDBOXED_SERVER_IDS`. It ships disabled.
