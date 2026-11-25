"""Runs INSIDE the CAD sandbox (cadquery, numpy, matplotlib) -- never imported by the MCP.

    python _probe.py run    --code answer.py [--render iso,t2 --size 512 --outdir .cad/render]
    python _probe.py step   --path part.step
Prints one JSON object on the LAST stdout line; everything else is the program's own output.
"""
from __future__ import annotations

import argparse, glob, io, json, math, os, runpy, sys, time, traceback
from contextlib import redirect_stdout, redirect_stderr

# cadquery 2.3 calls TopoDS_*.HashCode, which cadquery-ocp 7.9 removed. The BenchCAD image patches it in
# sitecustomize; do the same here so a plain local Python with the pinned versions works too.
try:
    from OCP.TopoDS import (TopoDS_Compound, TopoDS_CompSolid, TopoDS_Edge, TopoDS_Face, TopoDS_Shape,
                            TopoDS_Shell, TopoDS_Solid, TopoDS_Vertex, TopoDS_Wire)
    for _c in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex, TopoDS_Wire, TopoDS_Shell,
               TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid):
        if not hasattr(_c, "HashCode"):
            _c.HashCode = lambda self, ub=2147483647: id(self) % ub
except Exception:
    pass

# Nominal tetrahedral cameras used for BenchCAD T3/T4 input views, in 2x2 order.
# Only (1,1,1) is exact in the real inputs; the other three are rotated 3-8 deg.
VIEWS = {
    "t1": (-1, -1, 1), "iso": (1, 1, 1), "t3": (-1, 1, -1), "t4": (1, -1, -1),
    "front": (0, -1, 0), "back": (0, 1, 0), "right": (1, 0, 0),
    "left": (-1, 0, 0), "top": (0, 0, 1), "bottom": (0, 0, -1),
}
DEFAULT_VIEWS = "t1,iso,t3,t4"


def _shape(obj):
    import cadquery as cq
    if isinstance(obj, cq.Workplane):
        vals = [v for v in obj.vals() if isinstance(v, cq.Shape)]
        if not vals:
            raise ValueError("`result` is a Workplane with no solid on the stack")
        return vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
    if isinstance(obj, cq.Shape):
        return obj
    raise TypeError(f"`result` must be a cadquery Workplane or Shape, got {type(obj).__name__}")


def _metrics(sh):
    bb = sh.BoundingBox()
    return {
        "bbox_min": [round(bb.xmin, 3), round(bb.ymin, 3), round(bb.zmin, 3)],
        "bbox_max": [round(bb.xmax, 3), round(bb.ymax, 3), round(bb.zmax, 3)],
        "size": [round(bb.xlen, 3), round(bb.ylen, 3), round(bb.zlen, 3)],
        "volume": round(sh.Volume(), 3),
        "area": round(sh.Area(), 3),
        "solids": len(sh.Solids()), "faces": len(sh.Faces()), "edges": len(sh.Edges()),
        "valid": bool(sh.isValid()),
        "topology": _betti(sh),
        "center_of_mass": [round(c, 3) for c in sh.Center().toTuple()],
    }


def _mesh_genus(sh):
    """Total genus of a closed triangle mesh: chi = V - E + F = 2C - 2g, with vertices merged across faces."""
    import numpy as np
    V, T = _mesh(sh)
    q = np.round(V / max(sh.BoundingBox().DiagonalLength * 1e-7, 1e-9)).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    T = inv.reshape(-1)[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    edges, uses = np.unique(E, axis=0, return_counts=True)
    if (uses % 2).any():
        return None                                     # not closed: no genus to read
    n_edges = int((uses // 2).sum())                    # an edge four faces share is two edges, one per sheet
    nv = int(np.unique(T).size)
    parent = np.arange(int(inv.max()) + 1)              # connected surface components
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = len({find(v) for v in np.unique(T)})
    chi = nv - n_edges + len(T)
    return None if (2 * comps - chi) % 2 else (2 * comps - chi) // 2


def _betti(sh):
    """[b0, b1, b2] the way BenchCAD's part score counts them: b0 pieces (solids that touch count once),
    b1 through-holes (genus, per shell by the Euler-Poincare formula on the B-rep, or on the mesh when
    the B-rep has degenerate points), b2 sealed cavities (inner shells). None when it cannot be read."""
    try:
        from OCP.TopExp import TopExp, TopExp_Explorer
        from OCP.TopTools import TopTools_IndexedMapOfShape
        from OCP.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE, TopAbs_SHELL
        from OCP.BRepClass3d import BRepClass3d
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape

        def count(shape, kind):
            m = TopTools_IndexedMapOfShape(); TopExp.MapShapes_s(shape, kind, m); return m.Extent()
        solids = [x for x in sh.Solids() if x.Volume() > 0]
        if not solids:
            return None
        b1 = b2 = 0
        brep_ok = True
        for sol in solids:
            outer = BRepClass3d.OuterShell_s(sol.wrapped)
            ex = TopExp_Explorer(sol.wrapped, TopAbs_SHELL)
            while ex.More():
                shell = ex.Current()
                v, e, f = count(shell, TopAbs_VERTEX), count(shell, TopAbs_EDGE), count(shell, TopAbs_FACE)
                loops = 0
                fx = TopExp_Explorer(shell, TopAbs_FACE)
                while fx.More():
                    loops += count(fx.Current(), TopAbs_WIRE); fx.Next()
                chi = v - e + 2 * f - loops            # V - E + F - (inner loops) = 2 - 2g on a closed shell
                if chi % 2 or chi > 2:
                    brep_ok = False
                else:
                    b1 += (2 - chi) // 2
                if not shell.IsSame(outer):
                    b2 += 1
                ex.Next()
        if not brep_ok:                                 # poles, collapsed edges: count on the mesh instead
            b1 = sum(g for g in (_mesh_genus(x) for x in solids) if g is not None) \
                if all(_mesh_genus(x) is not None for x in solids) else None
            if b1 is None:
                return None
        # pieces: solids that touch or overlap are one piece
        parent = list(range(len(solids)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        tol = max(1e-6, 1e-7 * sh.BoundingBox().DiagonalLength)
        boxes = [x.BoundingBox() for x in solids]
        for i in range(len(solids)):
            for j in range(i + 1, len(solids)):
                a, b = boxes[i], boxes[j]
                if a.xmin > b.xmax + tol or b.xmin > a.xmax + tol or a.ymin > b.ymax + tol or \
                        b.ymin > a.ymax + tol or a.zmin > b.zmax + tol or b.zmin > a.zmax + tol:
                    continue
                d = BRepExtrema_DistShapeShape(solids[i].wrapped, solids[j].wrapped)
                if d.IsDone() and d.Value() <= tol:
                    parent[find(i)] = find(j)
        return [len({find(i) for i in range(len(solids))}), b1, b2]
    except Exception:
        return None


def _mesh(sh):
    import numpy as np
    diag = max(sh.BoundingBox().DiagonalLength, 1e-6)
    v, t = sh.tessellate(diag * 1e-3, 0.3)
    V = np.array([[p.x, p.y, p.z] for p in v], float)
    return V, np.array(t, int)


def _render(sh, names, size, outdir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V, T = _mesh(sh)
    tri = V[T]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    c = (V.max(0) + V.min(0)) / 2
    r = float((V.max(0) - V.min(0)).max()) / 2 * 1.05 or 1.0
    cols = int(math.ceil(math.sqrt(len(names)))); rows = int(math.ceil(len(names) / cols))
    fig = plt.figure(figsize=(cols * size / 100, rows * size / 100), dpi=100)
    for i, name in enumerate(names):
        d = np.array(VIEWS[name], float); d /= np.linalg.norm(d)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d", proj_type="ortho")
        up0 = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
        right = np.cross(up0, d); right /= np.linalg.norm(right)
        up = np.cross(d, right)
        light = 0.6 * d + 0.55 * up + 0.45 * right; light /= np.linalg.norm(light)
        nf = n * np.where(n @ d < 0, -1.0, 1.0)[:, None]   # winding-independent: face the camera
        shade = 0.22 + 0.78 * np.clip(nf @ light, 0, 1)
        rgb = np.outer(shade, np.array([0.18, 0.62, 0.62])).clip(0, 1)
        pc = Poly3DCollection(tri, facecolors=rgb, edgecolors="none", linewidths=0)
        ax.add_collection3d(pc)
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=math.degrees(math.asin(d[2])), azim=math.degrees(math.atan2(d[1], d[0])))
        ax.set_axis_off(); ax.set_title(name, fontsize=9)
    fig.subplots_adjust(0, 0, 1, 0.97, 0, 0)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "render.png")
    fig.savefig(path, dpi=100); plt.close(fig)
    return path


def cmd_run(a):
    buf = io.StringIO()
    out = {"ok": False}
    ns = {}
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            ns = runpy.run_path(a.code, run_name="__main__")
    except BaseException:
        out["error"] = "".join(traceback.format_exc().splitlines(True)[-14:])
    out["output_tail"] = buf.getvalue()[-4000:]
    if "error" not in out:
        if isinstance(ns.get("result"), dict):          # T6: a graph, not a solid
            g = ns["result"]
            out["ok"] = True
            out["note"] = ("result is a graph dict: "
                           + ", ".join(f"{k}={len(v)}" for k, v in g.items() if isinstance(v, list))
                           + ". Check it with netlist_validate after tools.export writes pred_graph.json.")
        elif "result" in ns:
            try:
                sh = _shape(ns["result"])
                out["metrics"] = _metrics(sh)
                m = out["metrics"]
                out["ok"] = m["valid"] and m["volume"] > 0 and m["solids"] >= 1
                if not m["valid"]: out["warning"] = "shape is not valid (self-intersection / bad topology)"
                elif m["volume"] <= 0: out["warning"] = "zero or negative volume"
                if a.render and out["ok"]:
                    names = [x for x in a.render.split(",") if x in VIEWS]
                    out["render_png"] = _render(sh, names, a.size, a.outdir)
            except BaseException:
                out["error"] = "".join(traceback.format_exc().splitlines(True)[-10:])
        else:
            sub = [os.path.join(dp, f) for dp, _, fs in os.walk("submission") for f in fs] if os.path.isdir("submission") else []
            out["files_in_submission"] = sub[:50]
            out["ok"] = bool(sub)
            out["note"] = "no `result` variable (fine for assembly tasks that write submission/)" if sub else "no `result` variable and nothing in submission/"
    print(json.dumps(out))


def _write_obj(sh, path, name):
    V, T = _mesh(sh)
    with open(path, "w") as f:
        f.write(f"o {name}\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in T:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def cmd_export(a):
    import cadquery as cq
    out = {"ok": False, "files": []}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            ns = runpy.run_path(a.code, run_name="__main__")
        if "result" not in ns:
            raise ValueError("the program leaves no `result`")
        sh = _shape(ns["result"])
        m = _metrics(sh)
        if not (m["valid"] and m["volume"] > 0):
            raise ValueError("`result` is not a valid solid with positive volume")
        os.makedirs(a.outdir, exist_ok=True)
        base = os.path.join(a.outdir, a.name)
        for fmt in [x for x in a.formats.split(",") if x]:
            path = f"{base}.{fmt}"
            if fmt == "step":
                cq.exporters.export(sh, path, exportType="STEP")
                back = _shape(cq.importers.importStep(path))      # round trip: re-read and compare
                out["step_roundtrip_volume_diff_pct"] = round(abs(back.Volume() - m["volume"]) / m["volume"] * 100, 4)
            elif fmt == "stl":
                cq.exporters.export(sh, path, exportType="STL", tolerance=a.tolerance, angularTolerance=0.1)
            elif fmt == "obj":
                _write_obj(sh, path, a.name)
            else:
                raise ValueError(f"unsupported format {fmt!r} (step, stl, obj)")
            out["files"].append(path)
        out.update(ok=True, metrics=m)
    except BaseException:
        out["error"] = "".join(traceback.format_exc().splitlines(True)[-10:])
    print(json.dumps(out))


def cmd_step(a):
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    sh = _shape(cq.importers.importStep(a.path))
    cyl, planes, other = [], 0, 0
    for f in sh.Faces():
        gt = f.geomType()
        if gt == "CYLINDER":
            s = BRepAdaptor_Surface(f.wrapped).Cylinder()
            ax, loc = s.Axis().Direction(), s.Location()
            cyl.append({"radius": round(s.Radius(), 3),
                        "axis": [round(ax.X(), 3), round(ax.Y(), 3), round(ax.Z(), 3)],
                        "axis_point": [round(loc.X(), 2), round(loc.Y(), 2), round(loc.Z(), 2)]})
        elif gt == "PLANE":
            planes += 1
        else:
            other += 1
    print(json.dumps({"metrics": _metrics(sh), "n_cylindrical": len(cyl), "n_planar": planes,
                      "n_other": other, "cylindrical_faces": cyl[:60]}))


def _basis(d):
    import numpy as np
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    up0 = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
    right = np.cross(up0, d); right /= np.linalg.norm(right)
    return d, right, np.cross(d, right)


def _silhouette(V, T, d, n_px, frame=None):
    """Filled-outline mask of the mesh seen along direction d (parallel projection), top = up.
    `frame`: vertices whose extent sets the scale and centre instead of the mesh's own, so several
    parts drawn with the same frame land where they are relative to one another."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    d, right, up = _basis(d)
    P = np.stack([V @ right, V @ up], axis=1)[T]
    ref = P if frame is None else np.stack([frame @ right, frame @ up], axis=1)
    lo, hi = ref.reshape(-1, 2).min(0), ref.reshape(-1, 2).max(0)
    half = float((hi - lo).max()) / 2 * 1.04 or 1.0
    mid = (lo + hi) / 2
    fig = plt.figure(figsize=(1, 1), dpi=n_px)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.add_collection(PolyCollection(P, facecolors="black", edgecolors="black", linewidths=0.3))
    ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half)
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., 0]
    plt.close(fig)
    return a < 128


def _perturbed(d, deg, az):
    """Camera direction d tilted by `deg` degrees towards azimuth `az` around d."""
    import numpy as np
    d, right, up = _basis(d)
    t = math.radians(deg)
    return math.cos(t) * d + math.sin(t) * (math.cos(az) * right + math.sin(az) * up)


def _fit(m, box, g):
    """Crop `m` to box=(y0,y1,x0,x1), fit into g x g keeping aspect, centred (nearest)."""
    import numpy as np
    y0, y1, x0, x1 = box
    c = m[y0:y1 + 1, x0:x1 + 1]
    h, w = c.shape
    k = (g - 2) / max(h, w)
    nh, nw = max(1, round(h * k)), max(1, round(w * k))
    yi = (np.arange(nh) / k).astype(int).clip(0, h - 1); xi = (np.arange(nw) / k).astype(int).clip(0, w - 1)
    out = np.zeros((g, g), m.dtype)
    y0o, x0o = (g - nh) // 2, (g - nw) // 2
    out[y0o:y0o + nh, x0o:x0o + nw] = c[np.ix_(yi, xi)]
    return out


def _box(m):
    import numpy as np
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def _norm_mask(m, g=160):
    import numpy as np
    b = _box(m)
    if b is None:
        return np.zeros((g, g), bool), 1.0
    return _fit(m, b, g), (b[3] - b[2] + 1) / (b[1] - b[0] + 1)


def _zbuffer(V, T, d, n_px):
    """Depth and flat-shaded image of the mesh along d. Returns (mask, shade, depth) at n_px x n_px."""
    import numpy as np
    d, right, up = _basis(d)
    P = np.stack([V @ right, V @ up], axis=1)
    lo, hi = P.min(0), P.max(0)
    half = float((hi - lo).max()) / 2 * 1.04 or 1.0
    mid = (lo + hi) / 2
    S = (P - (mid - half)) / (2 * half) * n_px            # pixel coords, y up
    S[:, 1] = n_px - S[:, 1]
    D = V @ d                                              # larger = nearer the camera
    zb = np.full((n_px, n_px), -np.inf); shade = np.zeros((n_px, n_px))
    light = _basis(d)[0] * 0.6 + up * 0.55 + right * 0.45; light /= np.linalg.norm(light)
    tri = V[T]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    nrm *= np.where(nrm @ d < 0, -1.0, 1.0)[:, None]
    lam = 0.25 + 0.75 * np.clip(nrm @ light, 0, 1)
    for k in range(len(T)):
        a, b, c = S[T[k]]
        x0, x1 = int(max(min(a[0], b[0], c[0]) - 1, 0)), int(min(max(a[0], b[0], c[0]) + 1, n_px - 1))
        y0, y1 = int(max(min(a[1], b[1], c[1]) - 1, 0)), int(min(max(a[1], b[1], c[1]) + 1, n_px - 1))
        if x1 < x0 or y1 < y0:
            continue
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / det
        l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / det
        l3 = 1 - l1 - l2
        inside = (l1 >= -0.02) & (l2 >= -0.02) & (l3 >= -0.02)
        z = l1 * D[T[k][0]] + l2 * D[T[k][1]] + l3 * D[T[k][2]]
        sub = zb[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (z > sub)
        sub[upd] = z[upd]
        shade[y0:y1 + 1, x0:x1 + 1][upd] = lam[k]
    mask = np.isfinite(zb)
    depth = np.where(mask, zb, 0.0)
    if mask.any():
        depth = np.where(mask, (zb - zb[mask].min()) / max(float(zb[mask].max() - zb[mask].min()), 1e-9), 0.0)
    return mask, shade, depth


def _edges(shade, depth, mask, interior_px=3):
    """Interior edges: shading steps and depth jumps, away from the outline."""
    import numpy as np

    def grad(a):
        gx = np.abs(np.diff(a, axis=1, prepend=a[:, :1])); gy = np.abs(np.diff(a, axis=0, prepend=a[:1, :]))
        return np.maximum(gx, gy)
    e = (grad(shade) > 0.06) | (grad(depth) > 0.04)
    m = mask.copy()
    for _ in range(interior_px):                                   # erode the mask: drop outline edges
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return e & m


def _dilate(a, r):
    import numpy as np
    o = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o |= np.roll(np.roll(a, dy, 0), dx, 1)
    return o


def _assembly_shape(root="submission", only=None, skip=None):
    """The assembly the scorer rebuilds: every instance in instances.json = its part STEP moved by its
    row-major 4x4 transform. `only`: just the instances of that part id; `skip`: all but those."""
    import cadquery as cq
    from OCP.gp import gp_Trsf
    recs = json.load(open(os.path.join(root, "assembly", "instances.json")))
    if isinstance(recs, dict):                           # the wrapped form: {"format", "instances": [...]}
        recs = recs.get("instances", [])
    parts, placed = {}, []
    for r in recs:
        pid = r["part_id"]
        if only is not None and pid != only:
            continue
        if skip is not None and pid == skip:
            continue
        if pid not in parts:
            parts[pid] = _shape(cq.importers.importStep(os.path.join(root, "parts", pid + ".step")))
        T = r.get("transform") or r["T"]
        R = [[float(T[i][j]) for j in range(3)] for i in range(3)]
        dev = max(abs(sum(R[i][k] * R[j][k] for k in range(3)) - (i == j)) for i in range(3) for j in range(3))
        if dev > 1e-6:      # the scorer rejects the instance (and so scores the part as missing)
            raise ValueError(f"instance {r.get('instance_id', pid)}: the 3x3 block is not a rotation "
                             f"(R R^T is off by {dev:.1e}; the scorer allows 1e-6). Build rotations from "
                             "exact cos/sin (math or numpy, no rounding of the entries) and do not hand-type "
                             "rounded decimals; no scale or shear.")
        tr = gp_Trsf()
        tr.SetValues(*[float(T[i][j]) for i in range(3) for j in range(4)])
        placed.append(parts[pid].moved(cq.Location(tr)))
    if not placed:
        raise ValueError(f"no instance of {only!r} in submission/assembly/instances.json")
    return cq.Compound.makeCompound(placed)


def _red_mask(q):
    """Pixels of the highlighted part in an `<id>_in_assembly.png` quadrant (red on grey and white)."""
    q = q.astype(float)
    return (q[..., 0] - (q[..., 1] + q[..., 2]) / 2) > 8     # also the pale red of a part seen through the shell


def _setup_view(q, name, Va, Ta, G, n_px):
    """Target masks and camera of one view of an `<id>_in_assembly.png` quadrant `q`: the whole assembly
    and the red part (both fitted to a G x G grid by the whole assembly's box), and the camera direction
    that makes the model's assembly (mesh Va, Ta, drawn at n_px) overlap the target's best. The real
    cameras are rotated 3-8 degrees from the nominal ones, hence the search. None if the part is not
    visible in this view."""
    import numpy as np
    whole_t = q.min(axis=2) < 245                      # everything drawn: the assembly
    red_t = _red_mask(q) & whole_t
    bt = _box(whole_t)
    if bt is None or not red_t.any():
        return None
    tn_all = _fit(whole_t, bt, G); tn_red = _fit(red_t, bt, G)
    best = (-1.0, 0.0, None, None)
    for deg, azs in ((0.0, [0.0]), (4.0, range(8)), (8.0, range(8))):
        for k in azs:
            dv = _perturbed(VIEWS[name], deg, k * math.pi / 4) if deg else np.array(VIEWS[name], float)
            cm = _silhouette(Va, Ta, dv, n_px)
            bc = _box(cm)
            if bc is None:
                continue
            cn = _fit(cm, bc, G)
            iou = float((tn_all & cn).sum()) / (float((tn_all | cn).sum()) or 1.0)
            if iou > best[0]:
                best = (iou, deg, dv, bc)
    asm_iou, deg, dv, bc = best
    if dv is None:
        return None
    return {"tn_all": tn_all, "tn_red": tn_red, "asm_iou": asm_iou, "deg": deg, "dv": dv, "bc": bc}


def cmd_placed(a):
    """Where is the part? Draw only the part's instances in the whole assembly's frame and compare with
    the red part in `<id>_in_assembly.png` (same four cameras as views.png)."""
    import numpy as np
    from PIL import Image
    pid = a.shape[len("placed:"):]
    whole, part = _assembly_shape(), _assembly_shape(only=pid)
    Va, Ta = _mesh(whole); Vp, Tp = _mesh(part)
    img = np.asarray(Image.open(a.target).convert("RGB"))
    H, W = img.shape[:2]
    names = [x for x in a.render.split(",") if x in VIEWS]
    G = 160
    res, panels = [], []
    for name, (r, c) in zip(names, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        q = img[r * H // 2:(r + 1) * H // 2, c * W // 2:(c + 1) * W // 2]
        S = q.shape[0]
        vw = _setup_view(q, name, Va, Ta, G, S)
        if vw is None:
            res.append({"view": name, "iou": 0.0, "note": "the part is not visible in this view of the input"})
            panels.append(np.full((G, G, 3), 255, np.uint8)); continue
        tn_all, tn_red, asm_iou, deg, dv, bc = (vw[k] for k in ("tn_all", "tn_red", "asm_iou", "deg", "dv", "bc"))
        pm = _silhouette(Vp, Tp, dv, S, frame=Va)          # the part, in the whole assembly's frame
        cn_part = _fit(pm, bc, G)
        inter = float((tn_red & cn_part).sum()); union = float((tn_red | cn_part).sum()) or 1.0
        def centre(m):
            ys, xs = np.where(m)
            return (xs.mean() / G, ys.mean() / G) if len(xs) else (0.5, 0.5)
        (tx, ty), (cx, cy) = centre(tn_red), centre(cn_part)
        res.append({"view": name, "iou": round(inter / union, 4), "assembly_iou": round(asm_iou, 4),
                    "camera_tilt_deg": deg,
                    "yours_vs_target": [round(cx - tx, 3), round(cy - ty, 3)],      # +x = right, +y = down, fraction of the view
                    "size_ratio": round(float(cn_part.sum()) / max(float(tn_red.sum()), 1.0), 2)})
        pnl = np.full((G, G, 3), 255, np.uint8)
        pnl[tn_all] = (225, 225, 225)
        pnl[tn_red & cn_part] = (120, 120, 120); pnl[cn_part & ~tn_red] = (230, 60, 60); pnl[tn_red & ~cn_part] = (60, 90, 230)
        panels.append(pnl)
    n = len(panels); canvas = np.full((2 * G, 2 * G, 3), 255, np.uint8)
    for i, p_ in enumerate(panels):
        canvas[(i // 2) * G:(i // 2 + 1) * G, (i % 2) * G:(i % 2 + 1) * G] = p_
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "compare.png")
    Image.fromarray(canvas).resize((canvas.shape[1] * 2, canvas.shape[0] * 2), Image.NEAREST).save(path)
    ious = [x["iou"] for x in res]
    print(json.dumps({"ok": True, "placed": pid, "views": res, "combined": round(sum(ious) / len(ious), 4),
                      "mean_iou": round(sum(ious) / len(ious), 4), "interior_f1": None, "compare_png": path,
                      "n_instances": len(part.Solids())}))


def cmd_placed_part(a):
    """`placed:<part>` without running any program: score what submission/ holds now."""
    a.shape = "placed:" + a.part
    cmd_placed(a)


def _compare_shape(ns, which):
    """`which`: auto (the program's `result`, else the submitted assembly), assembly, or part:<id>."""
    import cadquery as cq
    if which.startswith("part:"):
        path = os.path.join("submission", "parts", which[5:] + ".step")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not written; the program must call tools.export_part(..., {which[5:]!r})")
        return _shape(cq.importers.importStep(path))
    if which == "assembly" or ("result" not in ns and os.path.isfile("submission/assembly/instances.json")):
        return _assembly_shape()
    if "result" not in ns:
        raise ValueError("no `result` and no submission/assembly/instances.json to compare")
    return _shape(ns["result"])


def _interior(V, T, dirv, tmask_w, tgray_edges, G, WORK):
    """Interior-edge F1 of a mesh against the target view at one camera."""
    cmask, cshade, cdepth = _zbuffer(V, T, dirv, WORK)
    cedges = _edges(cshade, cdepth, cmask)
    tb, cb = _box(tmask_w), _box(cmask)
    if tb is None or cb is None:
        return None, None, None
    te = _fit(_dilate(tgray_edges, 1), tb, G) & _fit(tmask_w, tb, G)
    ce = _fit(_dilate(cedges, 1), cb, G) & _fit(cmask, cb, G)
    nt, nc = int(te.sum()), int(ce.sum())
    if nt + nc < 60:
        return None, te, ce
    p = float((ce & _dilate(te, 3)).sum()) / max(nc, 1)
    rr = float((te & _dilate(ce, 3)).sum()) / max(nt, 1)
    return 2 * p * rr / max(p + rr, 1e-9), te, ce


def _combined(ious, f1s):
    mean = sum(ious) / len(ious)
    fs = [f for f in f1s if f is not None]
    return mean if not fs else 0.5 * mean + 0.5 * sum(fs) / len(fs)


def _standins(sh, views, G, WORK):
    """The same score for plain stand-ins the size of the candidate's bounding box: a box and an
    elliptic cylinder along each axis. A model that does no better than these has the main shape wrong."""
    import cadquery as cq
    bb = sh.BoundingBox()
    L = [max(bb.xlen, 1e-3), max(bb.ylen, 1e-3), max(bb.zlen, 1e-3)]
    shapes = {"box": cq.Workplane("XY").box(*L).val()}
    for ax, plane, (a_, b_, h) in (("x", "YZ", (L[1], L[2], L[0])), ("y", "XZ", (L[0], L[2], L[1])),
                                   ("z", "XY", (L[0], L[1], L[2]))):
        e = cq.Workplane(plane).ellipse(a_ / 2, b_ / 2) if abs(a_ - b_) > 1e-6 else cq.Workplane(plane).circle(a_ / 2)
        shapes["cylinder along " + ax] = e.extrude(h).val()
    out = {}
    for nm, shp in shapes.items():
        V, T = _mesh(shp)
        ious, f1s = [], []
        for n_px, tn, tmask_w, tge, dirv in views:
            cn, _ = _norm_mask(_silhouette(V, T, dirv, n_px), G)
            ious.append(float((tn & cn).sum()) / (float((tn | cn).sum()) or 1.0))
            f1s.append(_interior(V, T, dirv, tmask_w, tge, G, WORK)[0])
        out[nm] = round(_combined(ious, f1s), 4)
    return out


def cmd_compare(a):
    """Run the program and compare it with a 2x2 four-view image: outline overlap (after a small camera
    search, since the real views are rotated 3-8 degrees) and an interior-edge match."""
    import numpy as np
    from PIL import Image
    with redirect_stdout(io.StringIO()):
        ns = runpy.run_path(a.code, run_name="__main__")
    if a.shape.startswith("placed:"):
        return cmd_placed(a)
    sh = _compare_shape(ns, a.shape)
    V, T = _mesh(sh)
    img = np.asarray(Image.open(a.target).convert("RGB")).astype(float)
    H, W = img.shape[:2]
    names = [x for x in a.render.split(",") if x in VIEWS]
    quad = [(0, 0), (0, 1), (1, 0), (1, 1)]
    G, WORK = 160, 320
    res, panels = [], []
    WORKMASKS = []
    for name, (r, c) in zip(names, quad):
        q = img[r * H // 2:(r + 1) * H // 2, c * W // 2:(c + 1) * W // 2]
        tm = q.min(axis=2) < 245
        tn, ta = _norm_mask(tm, G)
        # camera search: nominal, then two rings of tilts (the real cameras are 3-8 degrees off)
        best = (-1.0, 0.0, 0.0, None, None, None)
        for deg, azs in ((0.0, [0.0]), (4.0, range(8)), (8.0, range(8))):
            for k in azs:
                dirv = _perturbed(VIEWS[name], deg, k * math.pi / 4) if deg else np.array(VIEWS[name], float)
                cm = _silhouette(V, T, dirv, q.shape[0])
                cn, ca = _norm_mask(cm, G)
                u = float((tn | cn).sum()) or 1.0
                iou = float((tn & cn).sum()) / u
                if iou > best[0]:
                    best = (iou, deg, k * 45.0, cn, dirv, ca)
        iou, deg, az, cn, dirv, ca = best
        gray = np.asarray(Image.fromarray(q.astype(np.uint8)).convert("L").resize((WORK, WORK), Image.BOX), float) / 255.0
        tmask_w = np.asarray(Image.fromarray((tm * 255).astype(np.uint8)).resize((WORK, WORK), Image.BOX)) > 127
        tgray_edges = _edges(gray, np.zeros_like(gray), tmask_w)
        WORKMASKS.append((q.shape[0], tn, tmask_w, tgray_edges, dirv))
        f1, te, ce = _interior(V, T, dirv, tmask_w, tgray_edges, G, WORK)
        res.append({"view": name, "iou": round(iou, 4), "interior_f1": None if f1 is None else round(f1, 4),
                    "camera_tilt_deg": deg, "aspect_target": round(ta, 3), "aspect_candidate": round(ca, 3)})
        pnl = np.full((G, G, 3), 255, np.uint8)
        pnl[tn & cn] = (190, 190, 190); pnl[cn & ~tn] = (230, 60, 60); pnl[tn & ~cn] = (60, 90, 230)
        if te is not None:
            both = te & _dilate(ce, 2)
            pnl[te & ~both] = (0, 40, 160); pnl[ce & ~_dilate(te, 2)] = (150, 0, 0); pnl[both] = (20, 20, 20)
        panels.append(pnl)
    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * G, cols * G, 3), 255, np.uint8)
    for i, p in enumerate(panels):
        canvas[(i // cols) * G:(i // cols + 1) * G, (i % cols) * G:(i % cols + 1) * G] = p
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "compare.png")
    Image.fromarray(canvas).resize((canvas.shape[1] * 2, canvas.shape[0] * 2), Image.NEAREST).save(path)
    mean = sum(x["iou"] for x in res) / len(res)
    fs = [x["interior_f1"] for x in res if x["interior_f1"] is not None]
    interior = sum(fs) / len(fs) if fs else None
    combined = mean if interior is None else 0.5 * mean + 0.5 * interior
    try:
        floor = _standins(sh, WORKMASKS, G, WORK)
    except Exception as exc:  # noqa: BLE001
        floor = {"error": repr(exc)[:200]}
    shared = None
    if a.shape.startswith("part:") and len(names) == 4:
        try:
            shared = _shared_check(sh, a.target, names)
        except Exception as exc:  # noqa: BLE001
            shared = {"error": repr(exc)[:200]}
    print(json.dumps({"ok": True, "views": res, "mean_iou": round(mean, 4),
                      "interior_f1": None if interior is None else round(interior, 4),
                      "combined": round(combined, 4), "compare_png": path, "metrics": _metrics(sh),
                      "standins": floor, "shared": shared}))


# ── auto-placement: move one part type's instances until they cover the red part in the input ──────────

def _orthonormal(R):
    """The rotation closest to R (exact to rounding: the scorer rejects R R^T off by more than 1e-6)."""
    import numpy as np
    U, _, Vt = np.linalg.svd(np.asarray(R, float))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _right_angle_rotations():
    """The 24 proper rotations that map the axes onto the axes."""
    import itertools
    import numpy as np
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (j, sg) in enumerate(zip(perm, signs)):
                R[i, j] = sg
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def _about_point(R, c):
    """4x4 that rotates by R around the point c."""
    import numpy as np
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(c, float) - R @ np.asarray(c, float)
    return M


def _box_blur(m, r):
    """Mean filter of half-width r over a 2-D array (r = 0: unchanged)."""
    import numpy as np
    a = np.asarray(m, float)
    if r <= 0:
        return a
    k = 2 * r + 1
    pad = np.pad(a, r, mode="constant")
    c = pad.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = a.shape[0]; w = a.shape[1]
    tot = c[k:k + n, k:k + w] - c[:n, k:k + w] - c[k:k + n, :w] + c[:n, :w]
    return tot / (k * k)


def _pattern_search(f, x0, steps, min_steps, deadline, max_evals):
    """Maximise f(x) by trying +/- one step along each coordinate, keeping any gain, and halving the steps
    when nothing helps. Needs no gradient (the score is made of rendered masks). Returns (x, f(x), evals)."""
    import numpy as np
    x = np.array(x0, float); fx = f(x); evals = 1
    steps = np.array(steps, float); lo = np.array(min_steps, float)
    while evals < max_evals and time.time() < deadline and (steps > lo).any():
        improved = False
        for i in range(len(x)):
            if steps[i] <= lo[i]:
                continue
            for sg in (1.0, -1.0):
                if evals >= max_evals or time.time() >= deadline:
                    break
                y = x.copy(); y[i] += sg * steps[i]
                fy = f(y); evals += 1
                if fy > fx + 1e-9:
                    x, fx, improved = y, fy, True
                    break
        if not improved:
            steps = np.where(steps > lo, steps / 2, steps)
    return x, fx, evals


def _load_instances(root="submission"):
    """(container, records): the records of instances.json and, when the file is the wrapped form, the dict holding them."""
    recs = json.load(open(os.path.join(root, "assembly", "instances.json")))
    if isinstance(recs, dict):
        return recs, recs.get("instances", [])
    return None, recs


def _matrix(rec):
    import numpy as np
    T = np.array(rec.get("transform") if rec.get("transform") is not None else rec["T"], float)
    if T.shape == (3, 4):
        T = np.vstack([T, [0, 0, 0, 1]])
    return T


# --- frame-first placement ------------------------------------------------------------------------------
# The four cameras are parallel projections that share one frame: every image is centred on the assembly's
# centre and all use one scale. In that frame a part's outline in a view is its outline at the origin shifted
# by the projection of its position, so for any orientation the position follows in closed form from where
# the outlines sit (their centres). What is left to search is the orientation of each part, plus the scale
# and the small tilts of the cameras, which every part shares and so are estimated once from a few parts.
FS = 176                        # side of the grids the placer works on, in pixels
EXACT_VIEWS = ("iso",)          # cameras the task says are not tilted
ASSEMBLY_ROTATIONS = None       # filled on first use


def _rotvec(w):
    import numpy as np
    w = np.asarray(w, float); th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _surface_points(V, T, n=30000):
    """n points spread over the mesh surface (area weighted), enough to draw its outline."""
    import numpy as np
    rng = np.random.default_rng(0)
    A = V[T]; area = np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1)
    if not float(area.sum()) > 0:
        raise ValueError("the part has no surface")
    i = rng.choice(len(T), n, p=area / area.sum()); u = rng.random((n, 2)); f = u.sum(1) > 1; u[f] = 1 - u[f]
    return A[i, 0] + u[:, :1] * (A[i, 1] - A[i, 0]) + u[:, 1:] * (A[i, 2] - A[i, 0])


def _closed(m):
    """Mask `m` with the small holes between drawn points closed (grow by 2 px, shrink back)."""
    d = _box_blur(m.astype(float), 2) > 0.02
    return ~(_box_blur((~d).astype(float), 2) > 0.02)


def _splat(Pt, axes, sc):
    """Outline mask (FS x FS) of the points `Pt` (assembly centre = origin) seen along a camera with image
    axes (right, up) at `sc` pixels per unit; image centre = origin, +y down."""
    import numpy as np
    r, u = axes
    x = np.rint(FS / 2 + sc * (Pt @ r)).astype(int); y = np.rint(FS / 2 - sc * (Pt @ u)).astype(int)
    ok = (x >= 0) & (x < FS) & (y >= 0) & (y < FS)
    m = np.zeros((FS, FS), bool); m[y[ok], x[ok]] = True
    return _closed(m)


def _centroid(m):
    import numpy as np
    ys, xs = np.where(m)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([FS / 2.0, FS / 2.0])


def _iou(a, b):
    return float((a & b).sum()) / (float((a | b).sum()) or 1.0)


def _shift(m, dx, dy):
    """`m` moved by whole pixels (dx right, dy down), empty where it came from outside."""
    import numpy as np
    out = np.zeros_like(m)
    if abs(dx) >= FS or abs(dy) >= FS:
        return out
    ys, yd = (slice(0, FS - dy), slice(dy, FS)) if dy >= 0 else (slice(-dy, FS), slice(0, FS + dy))
    xs, xd = (slice(0, FS - dx), slice(dx, FS)) if dx >= 0 else (slice(-dx, FS), slice(0, FS + dx))
    out[yd, xd] = m[ys, xs]
    return out


def _red_targets(img, names, thr=4.0):
    """The red part's mask in each quadrant of an `<id>_in_assembly.png`, resized to FS x FS (None: not visible)."""
    import numpy as np
    from PIL import Image
    H, W = img.shape[:2]; out = []
    for name, (r, c) in zip(names, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        q = img[r * H // 2:(r + 1) * H // 2, c * W // 2:(c + 1) * W // 2].astype(float)
        red = ((q[..., 0] - (q[..., 1] + q[..., 2]) / 2) > thr) & (q.min(axis=2) < 245)
        m = np.asarray(Image.fromarray((red * 255).astype(np.uint8)).resize((FS, FS), Image.BILINEAR)) > 127
        out.append(m if int(m.sum()) >= 12 else None)
    return out


def _frame_axes(names, frame):
    """Image axes (right, up) of each camera, turned from its nominal pose by the frame's tilt vectors."""
    import numpy as np
    out = []
    for name, w in zip(names, frame["w"]):
        _, right, up = _basis(VIEWS[name]); Rw = _rotvec(w)
        out.append((Rw @ right, Rw @ up))
    return out


def _position(part, R, frame, axes):
    """Where the part (turned by R) must be so its outlines are centred on the red ones: least squares over
    the views. Returns the centre position, or None if fewer than two views see it."""
    import numpy as np
    A, b = [], []
    for ax, ct in zip(axes, part["tg"]):
        if ct is None:
            continue
        d = ct - _centroid(_splat(part["pts"] @ R.T, ax, frame["sc"]))
        A += [frame["sc"] * ax[0], -frame["sc"] * ax[1]]; b += list(d)
    if len(b) < 4:
        return None
    return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]


def _pose_iou(part, R, t, frame, axes):
    """Mean over the views of IoU between the part's outline at (R, t) and the red mask."""
    import numpy as np
    vals = [_iou(_splat(part["pts"] @ R.T + t, ax, frame["sc"]), tm)
            for ax, tm in zip(axes, part["tgt"]) if tm is not None]
    return float(np.mean(vals)) if vals else 0.0


def _orientations(fine=False):
    """The turns to try: each of the 24 right-angle orientations, turned about the vertical in 30 degree steps."""
    global ASSEMBLY_ROTATIONS
    if ASSEMBLY_ROTATIONS is None:
        import numpy as np
        ASSEMBLY_ROTATIONS = [_rot_z(y) @ Rc for Rc in [np.eye(3)] + _right_angle_rotations() for y in range(0, 360, 30)]
    return ASSEMBLY_ROTATIONS


def _best_single(part, frame, axes, rots, deadline, keep=3):
    """The best `keep` (iou, R, t) over the turns `rots`, each at its closed-form position."""
    res = []
    for R in rots:
        if time.time() > deadline and res:
            break
        t = _position(part, R, frame, axes)
        if t is not None:
            res.append((_pose_iou(part, R, t, frame, axes), R, t))
    res.sort(key=lambda x: -x[0])
    return res[:keep]


def _refine_turn(part, R0, frame, axes, deadline):
    """Fine search of the turn (three small angles); the position is re-solved for every try."""
    import numpy as np
    def f(x):
        R = _rotvec(x) @ R0; t = _position(part, R, frame, axes)
        return -1.0 if t is None else _pose_iou(part, R, t, frame, axes)
    x, fx, _ = _pattern_search(f, np.zeros(3), [0.14] * 3, [0.01] * 3, deadline, 120)
    R = _rotvec(x) @ R0
    return fx, R, _position(part, R, frame, axes)


def _solve_single(part, frame, axes, deadline):
    import numpy as np
    top = _best_single(part, frame, axes, _orientations(), deadline)
    if not top:
        return None
    best = None
    for k, (_, R, _t) in enumerate(top):
        r = _refine_turn(part, R, frame, axes, time.time() + max(0.5, (deadline - time.time()) / (len(top) - k)))
        if r[2] is not None and (best is None or r[0] > best[0]):
            best = r
    return [(best[0], best[1], best[2])] if best else None


def _group_poses(R0, p0, n):
    """Poses of n instances spaced evenly about the vertical axis through the origin: instance k is
    instance 0 (turn R0, position p0) turned 360*k/n degrees."""
    out = []
    for k in range(n):
        Rk = _rot_z(360.0 * k / n)
        out.append((Rk @ R0, Rk @ p0))
    return out


def _group_fit(part, poses, frame, axes):
    import numpy as np
    vals = []
    for ax, tm in zip(axes, part["tgt"]):
        if tm is None:
            continue
        pts = np.vstack([part["pts"] @ R.T + t for R, t in poses])
        vals.append(_iou(_splat(pts, ax, frame["sc"]), tm))
    return float(np.mean(vals)) if vals else 0.0


def _solve_group(part, frame, axes, n, reach, deadline):
    """n equal instances about the vertical axis: search the turn of instance 0 and its radius, phase and
    height. Outlines of the other instances are the same outline turned, so tries are cheap shifts."""
    import numpy as np
    sc = frame["sc"]
    views = [(v, ax, tm) for v, (ax, tm) in enumerate(zip(axes, part["tgt"])) if tm is not None]
    if len(views) < 2:
        return None
    ct = {v: _centroid(tm) for v, _, tm in views}
    radii = np.linspace(0.0, reach, 10); phases = np.arange(0, 360, 15.0)
    cands = []
    rots = [_rot_z(y) @ Rc for Rc in [np.eye(3)] + _right_angle_rotations() for y in range(0, 360, 60)]
    for R0 in rots:
        if time.time() > deadline and cands:
            break
        Rk = [_rot_z(360.0 * k / n) @ R0 for k in range(n)]
        base = {v: [_splat(part["pts"] @ R.T, ax, sc) for R in Rk] for v, ax, _ in views}
        cb = {v: np.mean([_centroid(m) for m in base[v]], axis=0) for v, _, _ in views}
        # the instances' positions cancel in the mean, so the union's centre fixes the height only
        num = den = 0.0
        for v, ax, _ in views:
            g = np.array([sc * ax[0][2], -sc * ax[1][2]]); d = ct[v] - cb[v]
            num += float(g @ d); den += float(g @ g)
        z = num / den if den > 1e-9 else 0.0
        for r in radii:
            for ph in phases:
                p0 = np.array([r * math.cos(math.radians(ph)), r * math.sin(math.radians(ph)), z])
                tot = 0.0
                for v, ax, tm in views:
                    u = np.zeros((FS, FS), bool)
                    for k in range(n):
                        pk = _rot_z(360.0 * k / n) @ p0
                        u |= _shift(base[v][k], int(round(sc * float(pk @ ax[0]))), int(round(-sc * float(pk @ ax[1]))))
                    tot += _iou(u, tm)
                cands.append((tot / len(views), R0, p0))
    cands.sort(key=lambda x: -x[0])
    picked = []
    for c in cands:                                            # distinct starting points
        if all(np.abs(c[2] - q[2]).max() > 0.05 * reach + 1e-6 or np.abs(c[1] - q[1]).max() > 0.2 for q in picked):
            picked.append(c)
        if len(picked) == 3:
            break
    best = None
    for k, (_, R0, p0) in enumerate(picked):
        def f(x):
            r, ph, z = x[0], x[1], x[2]
            q = np.array([r * math.cos(math.radians(ph)), r * math.sin(math.radians(ph)), z])
            return _group_fit(part, _group_poses(_rotvec(x[3:]) @ R0, q, n), frame, axes)
        ph0 = math.degrees(math.atan2(p0[1], p0[0])); r0 = float(np.hypot(p0[0], p0[1]))
        x, fx, _ = _pattern_search(f, np.array([r0, ph0, p0[2], 0, 0, 0]),
                                   [0.1 * reach, 10.0, 0.05 * reach, 0.14, 0.14, 0.14],
                                   [0.004 * reach, 0.4, 0.002 * reach, 0.01, 0.01, 0.01],
                                   time.time() + max(0.5, (deadline - time.time()) / (len(picked) - k)), 150)
        if best is None or fx > best[0]:
            q = np.array([x[0] * math.cos(math.radians(x[1])), x[0] * math.sin(math.radians(x[1])), x[2]])
            best = (fx, _group_poses(_rotvec(x[3:]) @ R0, q, n))
    return best


def _estimate_frame(parts, names, deadline):
    """Scale and camera tilts shared by every view, from the parts that are easiest to place (single, large)."""
    import numpy as np, itertools
    cand = [p for p in parts if p["n"] == 1] or parts
    cand = sorted(cand, key=lambda p: -sum(int(m.sum()) for m in p["tgt"] if m is not None))[:4]
    first = cand[0]
    px = max(max(np.ptp(np.where(m)[0]), np.ptp(np.where(m)[1])) for m in first["tgt"] if m is not None) + 1.0
    ext = float(np.ptp(first["pts"], axis=0).max()) or 1.0
    frame = {"sc": px / ext, "w": [np.zeros(3) for _ in names]}
    axes = _frame_axes(names, frame)
    quick = [np.eye(3)] + _right_angle_rotations()
    best = (-1.0, frame["sc"])
    for f in np.geomspace(0.6, 1.8, 13):                       # scale first, at the right-angle orientations only
        fr = {"sc": frame["sc"] * f, "w": frame["w"]}
        top = _best_single(first, fr, axes, quick, deadline, keep=1)
        if top and top[0][0] > best[0]:
            best = (top[0][0], fr["sc"])
    frame["sc"] = best[1]
    span = max(1.0, (deadline - time.time()))
    def turns():
        return [_best_single(p, frame, _frame_axes(names, frame), _orientations(), time.time() + span / 8, keep=1)[0][1]
                for p in cand]
    def total(fr, Rs):
        ax = _frame_axes(names, fr); vals = []
        for p, R in zip(cand, Rs):
            t = _position(p, R, fr, ax)
            if t is not None:
                vals.append(_pose_iou(p, R, t, fr, ax))
        return float(np.mean(vals)) if vals else 0.0
    Rs = turns()
    loose = [i for i, n in enumerate(names) if n not in EXACT_VIEWS]
    cur = total(frame, Rs)
    for i in loose:                                            # coarse grid of tilts, one camera at a time
        for w in itertools.product((-0.14, -0.07, 0.0, 0.07, 0.14), repeat=3):
            if np.linalg.norm(w) > 0.16 or time.time() > deadline:
                continue
            fr = {"sc": frame["sc"], "w": [x.copy() for x in frame["w"]]}; fr["w"][i] = np.array(w)
            v = total(fr, Rs)
            if v > cur:
                frame, cur = fr, v
    Rs = turns()
    for _ in range(3):
        cur = total(frame, Rs)
        for step in (0.06, 0.03, 0.015):
            for ax_i in range(3):
                for i in loose:
                    for sg in (1, -1):
                        fr = {"sc": frame["sc"], "w": [x.copy() for x in frame["w"]]}; fr["w"][i][ax_i] += sg * step
                        v = total(fr, Rs)
                        if v > cur:
                            frame, cur = fr, v
            for f in (0.97, 1.03):
                fr = {"sc": frame["sc"] * f, "w": frame["w"]}
                v = total(fr, Rs)
                if v > cur:
                    frame, cur = fr, v
        if time.time() > deadline:
            break
        Rs = turns()
    return {"sc": float(frame["sc"]), "w": [[float(c) for c in w] for w in frame["w"]], "anchor_iou": round(cur, 4)}


def _bom_counts():
    """{part_id: quantity} from bom.json, else from instances.json."""
    out = {}
    try:
        for it in json.load(open("bom.json")).get("items", []):
            out[it["part_id"]] = int(it.get("quantity", 1))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    if not out:
        try:
            for r in _load_instances()[1]:
                out[r["part_id"]] = out.get(r["part_id"], 0) + 1
        except (OSError, ValueError, KeyError):
            pass
    return out


def cmd_autoplace(a):
    """Place every part type (`--part all`) or one (`--part <id>`, with the frame saved by an earlier run in
    `--frame`) from the red outlines in `parts/<id>_in_assembly.png`. See the section comment: one frame
    (scale, camera tilts) is estimated from a few parts; each part's position then follows from its
    outline centres and only its orientation is searched. Prints JSON (frame, per-part transforms, fit);
    writes only the comparison picture."""
    import numpy as np
    import cadquery as cq
    from PIL import Image
    t0 = time.time()
    names = [x for x in a.render.split(",") if x in VIEWS][:4]
    if len(names) < 2:
        raise ValueError("need at least two views")
    counts = _bom_counts()
    if a.part == "all":
        ids = [os.path.basename(p)[:-5] for p in sorted(glob.glob("submission/parts/*.step"))]
        ids = [i for i in ids if os.path.isfile(f"parts/{i}_in_assembly.png")]
    else:
        ids = [a.part]
    if not ids:
        raise FileNotFoundError("no exported part with a parts/<id>_in_assembly.png: build and export the parts first")
    parts = []
    for pid in ids:
        step = f"submission/parts/{pid}.step"
        if not os.path.isfile(step):
            raise FileNotFoundError(f"{step} not found: build and export the part first")
        V, T = _mesh(_shape(cq.importers.importStep(step)))
        cen0 = (V.min(0) + V.max(0)) / 2
        img = np.asarray(Image.open(f"parts/{pid}_in_assembly.png").convert("RGB"))
        tgt = _red_targets(img, names)
        if sum(m is not None for m in tgt) < 2:
            continue
        n = max(1, (a.count if a.part != "all" and a.count else counts.get(pid, 1)))
        parts.append({"pid": pid, "pts": _surface_points(V, T) - cen0, "cen0": cen0, "n": n, "tgt": tgt,
                      "tg": [None if m is None else _centroid(m) for m in tgt]})
    if not parts:
        raise ValueError("the part is not visible (no red) in at least two views of its input image")
    total = max(20.0, a.seconds)
    if a.frame and os.path.isfile(a.frame):
        frame = json.load(open(a.frame))
    else:
        frame = _estimate_frame(parts, names, t0 + 0.5 * total)
    frame_f = {"sc": frame["sc"], "w": [np.array(w) for w in frame["w"]]}
    axes = _frame_axes(names, frame_f)
    reach = 0.45 * FS / frame_f["sc"]
    results, left = {}, list(parts)
    while left:
        part = left.pop(0)
        deadline = time.time() + max(2.0, (t0 + total - time.time()) / (len(left) + 1))
        if part["n"] == 1:
            sol = _solve_single(part, frame_f, axes, deadline)
            poses = None if sol is None else [(sol[0][1], sol[0][2])]
            fit = None if sol is None else sol[0][0]
        else:
            g = _solve_group(part, frame_f, axes, part["n"], reach, deadline)
            poses, fit = (None, None) if g is None else (g[1], g[0])
        if poses is None:
            continue
        mats = []
        for R, t in poses:
            M = np.eye(4); M[:3, :3] = _orthonormal(R); M[:3, 3] = t - M[:3, :3] @ part["cen0"]
            mats.append(M)
        results[part["pid"]] = {"n_instances": part["n"], "iou": round(float(fit), 4),
                                "transforms": [M.tolist() for M in mats], "_poses": poses, "_part": part}
    if not results:
        raise ValueError("no part could be placed")
    worst = min(results, key=lambda k: results[k]["iou"])
    part, poses = results[worst]["_part"], results[worst]["_poses"]
    panels = []
    for ax, tm in zip(axes, part["tgt"]):
        pnl = np.full((FS, FS, 3), 255, np.uint8)
        if tm is not None:
            m = _splat(np.vstack([part["pts"] @ R.T + t for R, t in poses]), ax, frame_f["sc"])
            pnl[tm & m] = (120, 120, 120); pnl[m & ~tm] = (230, 60, 60); pnl[tm & ~m] = (60, 90, 230)
        panels.append(pnl)
    while len(panels) < 4:
        panels.append(np.full((FS, FS, 3), 255, np.uint8))
    canvas = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:4])])
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "compare.png")
    Image.fromarray(canvas).resize((canvas.shape[1] * 2, canvas.shape[0] * 2), Image.NEAREST).save(path)
    for r in results.values():
        r.pop("_poses"); r.pop("_part")
    print(json.dumps({"ok": True, "autoplaced": a.part, "frame": frame, "parts": results, "worst": worst,
                      "seconds": round(time.time() - t0, 1), "compare_png": path}))


def _rot_z(deg):
    import numpy as np
    t = math.radians(deg); c, s_ = math.cos(t), math.sin(t)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


# --- shape check at one shared scale ---------------------------------------------------------------------
# BenchCAD scores a part's shape in 3-D, on its own bounding box, and only for what it gets above the best
# plain box / sphere / cylinder (a plain block scores 0). The per-view compare above rescales every view on
# its own, so a part with the wrong proportions can still look close. The four panels of parts/<id>_alone.png
# share one scale (the part drawn turned as it sits in the assembly), so here the part is drawn in all four
# at ONE scale and the best turn, and compared with plain blocks fitted the same way (their proportions
# searched too). `gain` = how much of the gap between the best plain block and a perfect match the part
# closes; 0 = no better than a plain block, which is what the benchmark gives such a part.
SHAPE_FLOOR_FILE = ".cad/shape_floor.json"


def _view_targets(img, n_views):
    """Mask of whatever is drawn (not white) in each quadrant, resized to FS x FS."""
    import numpy as np
    from PIL import Image
    H, W = img.shape[:2]; out = []
    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)][:n_views]:
        q = img[r * H // 2:(r + 1) * H // 2, c * W // 2:(c + 1) * W // 2].astype(float)
        m = q.min(axis=2) < 245
        out.append(np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((FS, FS), Image.BILINEAR)) > 127)
    return out


def _scale_fit(Q, tgt, axes):
    """Mean IoU of the points Q against the masks, all views at ONE scale (from the extents) and each
    outline centred on the target's (where it sits in the panel does not matter)."""
    import numpy as np
    ratios = []
    for ax, tm in zip(axes, tgt):
        ys, xs = np.where(tm)
        if len(xs) == 0:
            continue
        e = [float(np.ptp(Q @ ax[0])), float(np.ptp(Q @ ax[1]))]
        ratios += [(np.ptp(xs) + 1) / max(e[0], 1e-9), (np.ptp(ys) + 1) / max(e[1], 1e-9)]
    if not ratios:
        return 0.0
    sc = float(np.median(ratios)) * 0.98
    vals = []
    for ax, tm in zip(axes, tgt):
        m = _splat(Q, ax, sc)
        d = _centroid(tm) - _centroid(m)
        vals.append(_iou(_shift(m, int(round(d[0])), int(round(d[1]))), tm))
    return float(np.mean(vals))


def _shape_fit(pts, tgt, axes, stretch, seconds):
    """Best (iou, R, S): turn R over the assembly orientations then refined; with `stretch`, per-axis
    factors S of the part's own axes searched as well."""
    import numpy as np
    t0 = time.time()
    rots = _orientations()
    best = max(((_scale_fit(pts @ R.T, tgt, axes), R) for R in rots), key=lambda x: x[0])
    R0 = best[1]
    # The search converges in about 5 sweeps' worth of evaluations (25-115 evals against the sweep's 24).
    # On a busy machine the sweep alone used up the wall-clock budget and the search was cut off, so the
    # same part scored differently from run to run and the plain-block floor came out too low. The budget
    # is therefore never shorter than what convergence needs at the speed measured now.
    deadline = max(t0 + seconds, time.time() + 6.0 * (time.time() - t0))
    n = 6 if stretch else 3
    def f(x):
        S = np.exp(x[3:]) if stretch else np.ones(3)
        return _scale_fit((pts * S) @ (_rotvec(x[:3]) @ R0).T, tgt, axes)
    x, fx, _ = _pattern_search(f, np.zeros(n), [0.14] * 3 + [0.3] * (n - 3), [0.01] * 3 + [0.02] * (n - 3),
                               deadline, 300)
    return fx, _rotvec(x[:3]) @ R0, (np.exp(x[3:]) if stretch else np.ones(3))


def _plain_blocks(size):
    import cadquery as cq
    L = [max(float(v), 1e-3) for v in size]
    shapes = {"box": cq.Workplane("XY").box(*L).val()}
    for ax, plane, (a_, b_, h) in (("x", "YZ", (L[1], L[2], L[0])), ("y", "XZ", (L[0], L[2], L[1])),
                                   ("z", "XY", (L[0], L[1], L[2]))):
        shapes["cylinder along " + ax] = cq.Workplane(plane).ellipse(a_ / 2, b_ / 2).extrude(h).val()
    return shapes


def _shape_floor(target, size, tgt, axes, seconds):
    """Best score of a plain block (box or cylinder, proportions searched) against this target. Kept per
    target in SHAPE_FLOOR_FILE as the best ever found, so the floor only rises; the search is started again
    from the part's proportions when they differ from every start tried before."""
    import numpy as np
    try:
        cache = json.load(open(SHAPE_FLOOR_FILE))
    except (OSError, ValueError):
        cache = {}
    ent = cache.get(target) or {"best": 0.0, "name": None, "starts": []}
    r = sorted(np.asarray(size, float) / max(max(size), 1e-9))
    if any(np.abs(np.array(s) - r).max() < 0.15 for s in ent["starts"]):
        return ent
    for name, shp in _plain_blocks(size).items():
        V, T = _mesh(shp)
        p = _surface_points(V, T, 12000)
        v = _shape_fit(p - (p.min(0) + p.max(0)) / 2, tgt, axes, True, seconds / 4)[0]
        if v > ent["best"]:
            ent["best"], ent["name"] = round(v, 4), name
    ent["starts"].append([round(float(x), 3) for x in r])
    cache[target] = ent
    try:
        os.makedirs(os.path.dirname(SHAPE_FLOOR_FILE), exist_ok=True)
        json.dump(cache, open(SHAPE_FLOOR_FILE, "w"))
    except OSError:
        pass
    return ent


def _shared_check(sh, target, names, seconds=24.0):
    """The shape check for a part against a four-panel image: see the section comment."""
    import numpy as np
    from PIL import Image
    img = np.asarray(Image.open(target).convert("RGB"))
    tgt = _view_targets(img, len(names))
    w = [np.zeros(3) for _ in names]
    try:
        fr = json.load(open(".cad/frame.json"))
        if len(fr.get("w", [])) == len(names):
            w = [np.array(x, float) for x in fr["w"]]
    except (OSError, ValueError):
        pass
    axes = _frame_axes(names, {"w": w})
    V, T = _mesh(sh)
    p = _surface_points(V, T, 20000); p = p - (p.min(0) + p.max(0)) / 2
    bb = sh.BoundingBox(); size = [bb.xlen, bb.ylen, bb.zlen]
    iou, R, _ = _shape_fit(p, tgt, axes, False, seconds * 0.25)
    iou_s, _, S = _shape_fit(p, tgt, axes, True, seconds * 0.3)
    floor = _shape_floor(target, size, tgt, axes, seconds * 0.45)
    fl = float(floor["best"])
    gain = max(0.0, min(1.0, (iou - fl) / max(1e-6, 1.0 - fl)))
    S = S / float(np.median(S))
    return {"shared_iou": round(iou, 4), "plain_block": round(fl, 4), "plain_block_name": floor["name"],
            "gain": round(gain, 4), "stretched_iou": round(max(iou, iou_s), 4),
            "stretch": [round(float(s), 2) for s in S]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--code", required=True); r.add_argument("--render", default="")
    r.add_argument("--size", type=int, default=512); r.add_argument("--outdir", default=".cad/render")
    c = sub.add_parser("compare"); c.add_argument("--code", required=True); c.add_argument("--target", required=True)
    c.add_argument("--render", default=DEFAULT_VIEWS); c.add_argument("--outdir", default=".cad/render")
    c.add_argument("--shape", default="auto")
    pl = sub.add_parser("placed"); pl.add_argument("--part", required=True); pl.add_argument("--target", required=True)
    pl.add_argument("--render", default=DEFAULT_VIEWS); pl.add_argument("--outdir", default=".cad/render")
    ap = sub.add_parser("autoplace"); ap.add_argument("--part", default="all"); ap.add_argument("--target", default="")
    ap.add_argument("--frame", default="")
    ap.add_argument("--count", type=int, default=0); ap.add_argument("--render", default=DEFAULT_VIEWS)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--outdir", default=".cad/render")
    s = sub.add_parser("step"); s.add_argument("--path", required=True)
    e = sub.add_parser("export"); e.add_argument("--code", required=True); e.add_argument("--formats", default="step")
    e.add_argument("--outdir", default="exports"); e.add_argument("--name", default="part")
    e.add_argument("--tolerance", type=float, default=0.01)
    a = p.parse_args()
    {"run": cmd_run, "step": cmd_step, "export": cmd_export, "compare": cmd_compare, "autoplace": cmd_autoplace, "placed": cmd_placed_part}[a.cmd](a)
