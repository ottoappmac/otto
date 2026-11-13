"""The design spec: what the agent says it will build, checked against what the program produced.

The agent writes ``spec.json`` in the work folder before it writes any CAD code, from the user's words
(and states what it assumed). ``cad_run`` and ``cad_export`` then compare each result with it, so a part
that comes out the wrong size, with a missing or invented through-hole, or in several pieces is flagged
at once, in numbers, whatever the design is. It needs no reference picture.

    {
      "units": "mm",
      "parts": [
        {"id": "stand", "file": "part.py",
         "size": [80, 50, null],          # overall bounding box x, y, z; null = not specified
         "size_tol": 0.05,                # relative tolerance (default 5 %)
         "axes": "fixed",                 # "fixed" (default): x, y, z as written; "any": order does not matter
         "holes": 0,                      # through-holes (a tube or ring open at both ends counts one)
         "pieces": 1,                     # separate pieces; touching solids count once (default 1)
         "solids": 3,                     # solid bodies in the result, touching or not (an assembly in one STEP)
         "voids": 0,                      # sealed internal cavities
         "volume": [20000, 60000]}        # mm^3, [min, max]
      ],
      "assumptions": ["lip is 5 mm tall"]
    }

Only what the spec states is checked. Pure Python: no CadQuery needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPEC_FILE = "spec.json"
MAX_BYTES = 65536
DEFAULT_TOL = 0.05
MIN_TOL_MM = 0.1            # absolute floor, so a 2 mm wall is not held to 0.1 mm of a percent
AXES = ("x", "y", "z")


class SpecError(ValueError):
    pass


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise SpecError(f"{what} must be a number, got {v!r}")
    return float(v)


def _count(v: Any, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise SpecError(f"{what} must be a whole number 0 or more, got {v!r}")
    return v


def parse(text: str) -> dict[str, Any]:
    """Validate the spec text; returns {"parts": [...], "assumptions": [...]} or raises SpecError."""
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise SpecError(f"not valid JSON ({exc})") from None
    if not isinstance(raw, dict):
        raise SpecError("the top level must be an object with a \"parts\" list")
    parts_raw = raw.get("parts")
    if not isinstance(parts_raw, list) or not parts_raw:
        raise SpecError("\"parts\" must be a non-empty list")
    parts, seen = [], set()
    for i, p in enumerate(parts_raw):
        where = f"parts[{i}]"
        if not isinstance(p, dict):
            raise SpecError(f"{where} must be an object")
        pid = p.get("id")
        if not isinstance(pid, str) or not pid.strip():
            raise SpecError(f"{where} needs an \"id\"")
        if pid in seen:
            raise SpecError(f"two parts are called {pid!r}")
        seen.add(pid)
        where = f"part {pid!r}"
        out: dict[str, Any] = {"id": pid}
        if p.get("file") is not None:
            if not isinstance(p["file"], str):
                raise SpecError(f"{where}: \"file\" must be a file name")
            out["file"] = p["file"]
        if p.get("size") is not None:
            sz = p["size"]
            if not isinstance(sz, list) or len(sz) != 3:
                raise SpecError(f"{where}: \"size\" must be [x, y, z] (null for any you do not know)")
            vals = [None if v is None else _num(v, f"{where}: size") for v in sz]
            if any(v is not None and v <= 0 for v in vals):
                raise SpecError(f"{where}: size values must be above 0")
            if all(v is None for v in vals):
                raise SpecError(f"{where}: size is all null; leave it out instead")
            out["size"] = vals
        tol = p.get("size_tol", DEFAULT_TOL)
        tol = _num(tol, f"{where}: size_tol")
        if not 0 < tol <= 1:
            raise SpecError(f"{where}: size_tol must be above 0 and at most 1 (0.05 = 5 %)")
        out["size_tol"] = tol
        axes = p.get("axes", "fixed")
        if axes not in ("fixed", "any"):
            raise SpecError(f"{where}: \"axes\" must be \"fixed\" or \"any\"")
        out["axes"] = axes
        for key in ("holes", "pieces", "voids", "solids"):
            if p.get(key) is not None:
                out[key] = _count(p[key], f"{where}: {key}")
        for key in ("pieces", "solids"):
            if out.get(key) == 0:
                raise SpecError(f"{where}: {key} must be at least 1")
        if p.get("volume") is not None:
            v = p["volume"]
            if not isinstance(v, list) or len(v) != 2:
                raise SpecError(f"{where}: \"volume\" must be [min, max] in mm^3")
            lo, hi = _num(v[0], f"{where}: volume"), _num(v[1], f"{where}: volume")
            if not 0 <= lo <= hi:
                raise SpecError(f"{where}: volume must be [min, max] with 0 <= min <= max")
            out["volume"] = [lo, hi]
        if not any(k in out for k in ("size", "holes", "pieces", "voids", "solids", "volume")):
            raise SpecError(f"{where}: state at least one of size, holes, pieces, voids, solids, volume")
        parts.append(out)
    assumptions = raw.get("assumptions") or []
    if not isinstance(assumptions, list):
        raise SpecError("\"assumptions\" must be a list of short sentences")
    return {"parts": parts, "assumptions": [str(a) for a in assumptions][:20]}


def load(work: Path) -> dict[str, Any] | None:
    """The spec in `work`, None when there is none. Raises SpecError when it is there but unusable."""
    f = work / SPEC_FILE
    if not f.is_file():
        return None
    try:
        if f.stat().st_size > MAX_BYTES:
            raise SpecError(f"{SPEC_FILE} is over {MAX_BYTES // 1024} KB")
        return parse(f.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise SpecError(f"cannot read {SPEC_FILE} ({exc})") from None


def part_for(spec: dict[str, Any], code_file: str) -> tuple[dict[str, Any] | None, str]:
    """The spec part this program builds. A part with "file" matches that file; a spec with one part and
    no "file" applies to whatever program is run. Returns (part, why-not)."""
    code = Path(code_file).as_posix()
    for p in spec["parts"]:
        if p.get("file") and Path(p["file"]).as_posix() == code:
            return p, ""
    loose = [p for p in spec["parts"] if not p.get("file")]
    if len(spec["parts"]) == 1 and loose:
        return loose[0], ""
    names = ", ".join(p["id"] + (f" ({p['file']})" if p.get("file") else " (no file)") for p in spec["parts"])
    return None, (f"{code_file} is not tied to a part in {SPEC_FILE} (parts: {names}); "
                  "add \"file\" to the part this program builds")


def _tol(want: float, rel: float) -> float:
    return max(rel * want, MIN_TOL_MM)


def _pct(got: float, want: float) -> str:
    return f"{(got - want) / want * 100:+.0f}%"


def check(part: dict[str, Any], m: dict[str, Any]) -> list[str]:
    """Differences between the spec part and the metrics of the built solid (`size`, `volume`, `solids`,
    `topology` as cad_run reports them). Empty = every stated figure holds."""
    bad: list[str] = []
    size = m.get("size")
    want = part.get("size")
    if want and size and len(size) == 3:
        rel = part["size_tol"]
        if part["axes"] == "any":
            known = sorted((w for w in want if w is not None), reverse=True)
            got = sorted(size, reverse=True)
            # compare the stated extents with the nearest unused built extents, largest first
            pool = list(got)
            for w in known:
                g = min(pool, key=lambda v: abs(v - w))
                pool.remove(g)
                if abs(g - w) > _tol(w, rel):
                    bad.append(f"a side of {g:g} mm where the spec says {w:g} ({_pct(g, w)})")
        else:
            for ax, w, g in zip(AXES, want, size):
                if w is not None and abs(g - w) > _tol(w, rel):
                    bad.append(f"size {ax}: {g:g} mm, spec says {w:g} ({_pct(g, w)}, "
                               f"{'too long' if g > w else 'too short'})")
            known = [w for w in want if w is not None]
            if bad and len(known) == 3 and sorted(size) and all(
                    abs(a - b) <= _tol(b, rel) for a, b in zip(sorted(size), sorted(known))):
                bad.append("the three sizes match in a different order: an axis is swapped. If the "
                           "orientation does not matter, set \"axes\": \"any\" in the spec")
    vol = part.get("volume")
    if vol and m.get("volume") is not None and not vol[0] <= m["volume"] <= vol[1]:
        bad.append(f"volume {m['volume']:g} mm3 is outside the spec's {vol[0]:g} to {vol[1]:g}")
    if "solids" in part and m.get("solids") is not None and m["solids"] != part["solids"]:
        bad.append(f"{m['solids']} solid body(ies), spec says {part['solids']}")
    pieces = part.get("pieces", 1 if (("solids" not in part) and (want or "holes" in part or "voids" in part)) else None)
    topo = m.get("topology")
    if pieces is not None:
        got_p = topo[0] if topo else m.get("solids")
        if got_p is not None and got_p != pieces:
            bad.append(f"{got_p} separate piece(s), spec says {pieces}"
                       + ("; join them (union) or make them touch" if got_p > pieces else ""))
    for key, idx, label in (("holes", 1, "through-hole(s)"), ("voids", 2, "sealed void(s)")):
        if key in part:
            if not topo:
                bad.append(f"cannot read the {label} count to check it against the spec's {part[key]}")
            elif topo[idx] != part[key]:
                bad.append(f"{topo[idx]} {label}, spec says {part[key]}"
                           + (" (a hole is missing)" if key == "holes" and topo[idx] < part[key]
                              else " (one too many)" if key == "holes" else ""))
    return bad


def lines(work: Path, code_file: str, metrics: dict[str, Any] | None) -> list[str]:
    """What cad_run / cad_export append: nothing when there is no spec.json, else OK or the mismatches."""
    try:
        spec = load(work)
    except SpecError as exc:
        return [f"SPEC PROBLEM: {SPEC_FILE} {exc}. Fix the file; its checks are off until then."]
    if spec is None or not metrics:
        return []
    part, why = part_for(spec, code_file)
    if part is None:
        return [f"SPEC: {why}."]
    bad = check(part, metrics)
    if not bad:
        stated = [k for k in ("size", "volume", "pieces", "solids", "holes", "voids") if k in part]
        return [f"SPEC OK ({part['id']}): matches {SPEC_FILE} on {', '.join(stated) or 'what it states'}."]
    return ([f"SPEC MISMATCH ({part['id']}): the result does not match {SPEC_FILE}:"]
            + [f"  - {b}" for b in bad]
            + ["  Fix the program, or, if the user changed the requirement or the spec was wrong, "
               "update spec.json and say so. Do not report the part as done while this stands."])


def assumptions(work: Path) -> list[str]:
    try:
        spec = load(work)
    except SpecError:
        return []
    return spec["assumptions"] if spec else []
