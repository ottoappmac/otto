#!/usr/bin/env python3
"""Generate macOS-style app icons (rounded squircle + glass rim) for OTTO.

Source glyph: app/src/assets/logo-dark.png (white orbit glyph on transparent).
Produces the full Tauri icon set, icon.icns and icon.ico.

Run:  ../../../.venv/bin/python generate_icons.py
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
GLYPH_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src", "assets", "logo-dark.png"))

SS = 4                     # supersample factor for smooth antialiasing
BASE = 1024               # logical master size
S = BASE * SS
MARGIN = int(0.085 * BASE) * SS   # transparent margin around the squircle
N_EXP = 5.0               # superellipse exponent (Apple-like squircle)

# brand background (near-black with a hint of depth)
BG_TOP = np.array([34, 34, 36], dtype=np.float64)
BG_BOTTOM = np.array([6, 6, 7], dtype=np.float64)


def squircle_alpha(a_half, cx, cy, sharp=80.0):
    """Soft-edged superellipse alpha (0..1) over the SxS canvas."""
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float64)
    nx = (xs - cx) / a_half
    ny = (ys - cy) / a_half
    val = np.abs(nx) ** N_EXP + np.abs(ny) ** N_EXP
    # smooth transition around val == 1
    alpha = np.clip((1.0 - val) * sharp + 0.5, 0.0, 1.0)
    return alpha


def build_master():
    cx = cy = S / 2.0
    a = (S - 2 * MARGIN) / 2.0

    body = squircle_alpha(a, cx, cy)            # full squircle alpha
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float64)

    # vertical 0..1 (top->bottom) within the body for gradients
    top = cy - a
    bottom = cy + a
    v = np.clip((ys - top) / (bottom - top), 0.0, 1.0)
    vg = v[..., None]

    # background gradient fill
    rgb = BG_TOP * (1.0 - vg) + BG_BOTTOM * vg

    # subtle top gloss highlight (broad soft sheen in upper half)
    gloss_cx = cx
    gloss_cy = cy - a * 0.55
    gloss = np.exp(-(((xs - gloss_cx) / (a * 1.05)) ** 2) - (((ys - gloss_cy) / (a * 0.62)) ** 2))
    gloss *= np.clip(1.0 - v * 1.4, 0.0, 1.0)
    rgb += (255.0 - rgb) * (gloss[..., None] * 0.10)

    # ---- glass rim ----
    stroke = a * 0.022                          # rim thickness
    inner = squircle_alpha(a - stroke, cx, cy)
    rim = np.clip(body - inner, 0.0, 1.0)       # the ring band

    # rim brightness: bright at the top, dimmer (and slightly dark) at the bottom
    rim_light = np.clip(1.0 - v * 0.95, 0.0, 1.0)          # 1 top -> ~0 bottom
    light_amt = (0.30 + 0.70 * rim_light) * rim            # white highlight
    rgb += (255.0 - rgb) * light_amt[..., None]

    # faint dark contact line at the very bottom rim for depth
    dark_amt = np.clip(v - 0.55, 0.0, 1.0) / 0.45 * rim * 0.25
    rgb *= (1.0 - dark_amt[..., None] * 0.6)

    rgb = np.clip(rgb, 0, 255)

    base = np.dstack([rgb, body * 255.0]).astype(np.uint8)
    img = Image.fromarray(base, "RGBA")

    # ---- glyph ----
    glyph = Image.open(GLYPH_SRC).convert("RGBA")
    g_target = int((S - 2 * MARGIN) * 0.66)
    glyph = glyph.resize((g_target, g_target), Image.LANCZOS)
    gx = int(cx - g_target / 2)
    gy = int(cy - g_target / 2 - a * 0.01)      # nudge up for optical centering
    img.alpha_composite(glyph, (gx, gy))

    # downsample to master size
    master = img.resize((BASE, BASE), Image.LANCZOS)
    return master


def add_drop_shadow(master):
    """Place the squircle on a transparent canvas with a soft drop shadow."""
    canvas = Image.new("RGBA", (BASE, BASE), (0, 0, 0, 0))
    alpha = master.split()[3]
    shadow = Image.new("RGBA", (BASE, BASE), (0, 0, 0, 0))
    sh = Image.new("RGBA", (BASE, BASE), (0, 0, 0, 110))
    shadow.paste(sh, (0, int(BASE * 0.012)), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(BASE * 0.012))
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(master)
    return canvas


def main():
    master = build_master()
    master = add_drop_shadow(master)
    master.save(os.path.join(HERE, "icon.png"))

    sizes = {
        "32x32.png": 32,
        "64x64.png": 64,
        "128x128.png": 128,
        "128x128@2x.png": 256,
        "Square30x30Logo.png": 30,
        "Square44x44Logo.png": 44,
        "Square71x71Logo.png": 71,
        "Square89x89Logo.png": 89,
        "Square107x107Logo.png": 107,
        "Square142x142Logo.png": 142,
        "Square150x150Logo.png": 150,
        "Square284x284Logo.png": 284,
        "Square310x310Logo.png": 310,
        "StoreLogo.png": 50,
    }
    for name, px in sizes.items():
        master.resize((px, px), Image.LANCZOS).save(os.path.join(HERE, name))

    # favicon
    fav = os.path.normpath(os.path.join(HERE, "..", "..", "public", "favicon.png"))
    master.resize((256, 256), Image.LANCZOS).save(fav)

    # .ico (multi-size)
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    master.save(os.path.join(HERE, "icon.ico"), sizes=ico_sizes)

    # .icns via iconutil
    build_icns(master)

    # preview at 512 for review
    master.resize((512, 512), Image.LANCZOS).save(os.path.join(HERE, "_preview.png"))
    print("done")


def build_icns(master):
    iconset = os.path.join(HERE, "icon.iconset")
    os.makedirs(iconset, exist_ok=True)
    spec = [
        (16, "icon_16x16.png"), (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"), (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"), (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"), (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"), (1024, "icon_512x512@2x.png"),
    ]
    for px, fn in spec:
        master.resize((px, px), Image.LANCZOS).save(os.path.join(iconset, fn))
    subprocess.run(
        ["iconutil", "-c", "icns", iconset, "-o", os.path.join(HERE, "icon.icns")],
        check=True,
    )
    # cleanup iconset
    for fn in os.listdir(iconset):
        os.remove(os.path.join(iconset, fn))
    os.rmdir(iconset)


if __name__ == "__main__":
    main()
