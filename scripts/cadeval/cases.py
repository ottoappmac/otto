"""The CAD eval set: prompts a user might type, each with figures fixed by hand and a reference program.

Every `expect` is the cad_spec format (backend/builtin_mcps/cad_tools/cad_spec.py). The numbers come from the
prompt's own dimensions and from arithmetic done here, NOT from the agent, so an agent cannot pass by writing
itself an easy spec. Each reference program is run by `eval.sh selftest`: it must pass its own check (this
validates the case and the checker), and a version with its top half cut off must fail (the checker is not
blind). Text prompts only so far: nothing here tests reading a drawing or a photo (BenchCAD covers inputs
like that).

categories: part (stated dimensions), assembly (several solid bodies in one file), open (loose prompt,
loose figures), hard (a feature that is easy to get wrong).
"""

from __future__ import annotations

from math import cos, pi, sqrt

PCT = 0.015                    # volume range: the stated figure +/- this share


def _vol(v: float, pct: float = PCT) -> list[float]:
    return [round(v * (1 - pct), 1), round(v * (1 + pct), 1)]


CASES: list[dict] = []


def case(id, category, prompt, expect, reference, loose=False):
    """`loose`: the figures are too weak to notice a part with its top cut off (height is not stated), so
    selftest does not demand that the broken copy fails."""
    CASES.append({"id": id, "category": category, "prompt": " ".join(prompt.split()),
                  "expect": expect, "reference": reference.strip() + "\n", "loose": loose})


# ── parts with stated dimensions ────────────────────────────────────────────

case("plate_corner_holes", "part",
     "Make a flat rectangular plate 100 mm long (x), 60 mm wide (y) and 5 mm thick (z) with a 6 mm through-hole "
     "in each corner, centred 10 mm in from both edges.",
     {"size": [100, 60, 5], "holes": 4, "pieces": 1,
      "volume": _vol(100 * 60 * 5 - 4 * pi * 3 ** 2 * 5)},
     """
import cadquery as cq
result = (cq.Workplane("XY").box(100, 60, 5).faces(">Z").workplane()
          .rect(80, 40, forConstruction=True).vertices().hole(6))
""")

case("spacer_ring", "part",
     "Make a round spacer: 30 mm outside diameter, 16 mm bore, 10 mm long, axis along z.",
     {"size": [30, 30, 10], "holes": 1, "pieces": 1, "volume": _vol(pi * (15 ** 2 - 8 ** 2) * 10)},
     """
import cadquery as cq
result = cq.Workplane("XY").circle(15).circle(8).extrude(10)
""")

case("l_bracket", "part",
     "Make an L-shaped bracket from 4 mm thick material. The base is 60 mm wide (x) and 40 mm deep (y). A vertical "
     "flange rises along the back edge (the y = 40 side) across the full 60 mm width, so the overall height is "
     "34 mm. Put two 5 mm holes through the base, anywhere that is clear of the flange.",
     {"size": [60, 40, 34], "holes": 2, "pieces": 1,
      "volume": _vol(60 * 40 * 4 + 60 * 4 * 30 - 2 * pi * 2.5 ** 2 * 4)},
     """
import cadquery as cq
base = cq.Workplane("XY").box(60, 40, 4, centered=(True, True, False))
wall = cq.Workplane("XY").box(60, 4, 34, centered=(True, True, False)).translate((0, 18, 0))
result = base.union(wall).faces("<Z").workplane().pushPoints([(-15, 0), (15, 0)]).hole(5)
""")

case("open_box", "part",
     "Make an open-top rectangular box, 60 mm long (x), 40 mm wide (y) and 25 mm tall (z), with 2 mm walls and a "
     "2 mm floor.",
     {"size": [60, 40, 25], "holes": 0, "voids": 0, "pieces": 1,
      "volume": _vol(60 * 40 * 25 - 56 * 36 * 23)},
     """
import cadquery as cq
result = cq.Workplane("XY").box(60, 40, 25, centered=(True, True, False)).faces(">Z").shell(-2)
""")

case("hex_bolt", "part",
     "Make a hex-head bolt: 13 mm across the flats, head 5.3 mm high, with a plain 8 mm diameter shank 30 mm long "
     "below the head (no thread), axis along z, all one solid.",
     {"size": [13 / cos(pi / 6), 13, 35.3], "axes": "any", "holes": 0, "pieces": 1,
      "volume": _vol((sqrt(3) / 2) * 13 ** 2 * 5.3 + pi * 4 ** 2 * 30, 0.02)},
     """
import cadquery as cq
from math import cos, pi
result = (cq.Workplane("XY").polygon(6, 13 / cos(pi / 6)).extrude(5.3)
          .faces("<Z").workplane().circle(4).extrude(30))
""")

case("flange", "part",
     "Make a round flange: 80 mm outside diameter, 8 mm thick, a 30 mm centre bore, and six 8 mm holes evenly "
     "spaced on a 60 mm bolt circle. Axis along z.",
     {"size": [80, 80, 8], "holes": 7, "pieces": 1,
      "volume": _vol(pi * (40 ** 2 - 15 ** 2) * 8 - 6 * pi * 4 ** 2 * 8)},
     """
import cadquery as cq
result = (cq.Workplane("XY").circle(40).extrude(8).faces(">Z").workplane().hole(30)
          .faces(">Z").workplane().polarArray(30, 0, 360, 6).hole(8))
""")

case("hole_grid_panel", "part",
     "Make a flat panel 120 mm (x) by 80 mm (y) by 6 mm thick with 10 mm through-holes in a grid of 3 columns by "
     "2 rows, 35 mm apart across and 40 mm apart down, centred on the panel.",
     {"size": [120, 80, 6], "holes": 6, "pieces": 1, "volume": _vol(120 * 80 * 6 - 6 * pi * 5 ** 2 * 6)},
     """
import cadquery as cq
result = cq.Workplane("XY").box(120, 80, 6).faces(">Z").workplane().rarray(35, 40, 3, 2).hole(10)
""")

case("square_tube", "part",
     "Make a square tube, 20 mm outside, 2 mm wall, 100 mm long, open at both ends, along z.",
     {"size": [20, 20, 100], "holes": 1, "pieces": 1, "volume": _vol((20 * 20 - 16 * 16) * 100)},
     """
import cadquery as cq
result = cq.Workplane("XY").rect(20, 20).rect(16, 16).extrude(100)
""")

case("stepped_shaft", "part",
     "Make a stepped shaft along z, one solid: 20 mm diameter for 40 mm, then 12 mm diameter for 30 mm, then 8 mm "
     "diameter for 20 mm, joined end to end (90 mm long in total).",
     {"size": [20, 20, 90], "holes": 0, "pieces": 1, "volume": _vol(pi * (10 ** 2 * 40 + 6 ** 2 * 30 + 4 ** 2 * 20))},
     """
import cadquery as cq
result = (cq.Workplane("XY").circle(10).extrude(40).faces(">Z").workplane().circle(6).extrude(30)
          .faces(">Z").workplane().circle(4).extrude(20))
""")

case("slotted_plate", "part",
     "Make a plate 80 mm (x) by 30 mm (y) by 6 mm (z) with one slot cut straight through it: 8 mm wide, 40 mm long "
     "overall including its rounded ends, centred, long side along x.",
     {"size": [80, 30, 6], "holes": 1, "pieces": 1,
      "volume": _vol((80 * 30 - ((40 - 8) * 8 + pi * 4 ** 2)) * 6)},
     """
import cadquery as cq
result = cq.Workplane("XY").box(80, 30, 6).faces(">Z").workplane().slot2D(40, 8).cutThruAll()
""")

case("u_channel", "part",
     "Make a U-channel 80 mm long along z. Its outer cross-section is 50 mm wide (x) by 30 mm tall (y), the wall is "
     "3 mm thick, and it is open on the +y side.",
     {"size": [50, 30, 80], "holes": 0, "pieces": 1, "volume": _vol((50 * 30 - 44 * 27) * 80)},
     """
import cadquery as cq
outer = cq.Workplane("XY").box(50, 30, 80)
inner = cq.Workplane("XY").box(44, 27, 80).translate((0, 1.5, 0))
result = outer.cut(inner)
""")

case("cup", "part",
     "Make a drinking cup: 80 mm outside diameter, 95 mm tall, 3 mm wall, 5 mm thick floor, open top, axis along z.",
     {"size": [80, 80, 95], "holes": 0, "voids": 0, "pieces": 1,
      "volume": _vol(pi * (40 ** 2 * 95 - 37 ** 2 * 90))},
     """
import cadquery as cq
result = cq.Workplane("XY").circle(40).extrude(95).faces(">Z").workplane().hole(74, 90)
""")

# ── several solid bodies in one file ────────────────────────────────────────

case("box_with_lid", "assembly",
     "Make a box with a separate lid, as two solid bodies in one file. The box is open-top, 60 mm (x) by 40 mm (y) "
     "by 20 mm (z) outside, with 2 mm walls and a 2 mm floor. The lid is a flat 60 x 40 x 2 mm plate resting on "
     "top of the box rim.",
     {"size": [60, 40, 22], "solids": 2, "volume": _vol(60 * 40 * 20 - 56 * 36 * 18 + 60 * 40 * 2)},
     """
import cadquery as cq
base = cq.Workplane("XY").box(60, 40, 20, centered=(True, True, False)).faces(">Z").shell(-2)
lid = cq.Workplane("XY").box(60, 40, 2, centered=(True, True, False)).translate((0, 0, 20))
result = cq.Workplane("XY").add(base.val()).add(lid.val())
""")

case("stacked_discs", "assembly",
     "Make three separate round discs stacked on top of each other along z, all centred on the axis: diameters "
     "40, 30 and 20 mm, each 5 mm thick, the largest at the bottom. Keep them as three separate solid bodies in "
     "one file.",
     {"size": [40, 40, 15], "solids": 3, "volume": _vol(pi * (20 ** 2 + 15 ** 2 + 10 ** 2) * 5)},
     """
import cadquery as cq
discs = [cq.Workplane("XY").circle(r).extrude(5).translate((0, 0, 5 * i)).val() for i, r in enumerate((20, 15, 10))]
result = cq.Workplane("XY").add(discs)
""")

# ── loose prompt, loose figures ─────────────────────────────────────────────

case("phone_stand", "open",
     "Make a simple phone stand, about 80 mm wide and 50 mm deep, that holds a phone leaning back at about 60 "
     "degrees, with a 5 mm lip at the front so the phone does not slide off.",
     {"size": [80, 50, None], "size_tol": 0.2, "pieces": 1, "holes": 0},
     """
import cadquery as cq
base = cq.Workplane("XY").box(80, 50, 5, centered=(True, True, False))
lip = cq.Workplane("XY").box(80, 4, 10, centered=(True, True, False)).translate((0, -23, 0))
back = (cq.Workplane("XY").box(80, 4, 40, centered=(True, True, False))
        .rotate((0, 0, 0), (1, 0, 0), -30).translate((0, 5, 5)))
result = base.union(lip).union(back)
""", loose=True)

# ── a feature that is easy to get wrong ─────────────────────────────────────

case("spur_gear", "hard",
     "Make a spur gear with 20 teeth, module 2 (so 44 mm outside diameter), 8 mm thick, with an 8 mm centre bore. "
     "The tooth shape can be simplified.",
     {"size": [44, 44, 8], "size_tol": 0.06, "holes": 1, "pieces": 1},
     """
import cadquery as cq
core = cq.Workplane("XY").circle(18).extrude(8)
teeth = cq.Workplane("XY").add([
    cq.Workplane("XY").box(4, 2.6, 8, centered=(False, True, False)).translate((18 - 1, 0, 0))
      .rotate((0, 0, 0), (0, 0, 1), 18 * i).val() for i in range(20)])
result = core.union(teeth).faces(">Z").workplane().hole(8)
""")


def by_id(ids: list[str] | None = None) -> list[dict]:
    if not ids:
        return list(CASES)
    want = set(ids)
    return [c for c in CASES if c["id"] in want or c["category"] in want]


if __name__ == "__main__":
    for c in CASES:
        print(f"{c['id']:20s} {c['category']:9s} {c['expect']}")
