"""Generic bar-chart-race renderer (Skia -> ffmpeg), v2. Everything topic-specific lives in config.json.

Usage (from the project root):
  python3 engine/render.py --config config.json --check            # print ranking-quality metrics only
  python3 engine/render.py --config config.json --preview          # save key frames to build/frames/
  python3 engine/render.py --config config.json --preview 600 1200 # specific frames
  python3 engine/render.py --config config.json                    # full silent video -> build/race_silent.mp4
  python3 engine/render.py --config config.json --start 0 --end 2000 --out build/chunk_0.mp4   # one chunk

Data CSV (long format): <period_col>,<code_col>,<series cols...>
  period = "YYYY-MM" (monthly) or "YYYY" (yearly); one row per entity per period.
  Empty cells = entity not present yet / any more (it slides in or out of the race smoothly).

v2 additions: empty cells (entities entering/leaving), dynamic pinned slots ("max" on top, "min" at the bottom,
cross-faded when they change), rank numbers, ratio label (series.ratio_to), lifted rows while moving,
year pulse, drifting bokeh background, configurable 3D depth.
"""
import argparse
import colorsys
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import skia
from scipy.interpolate import PchipInterpolator

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.environ.get("CONFIG", "config.json"))
ap.add_argument("--preview", type=int, nargs="*")
ap.add_argument("--check", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=None)
ARGS, _ = ap.parse_known_args()

CFG_PATH = Path(ARGS.config).resolve()
ROOT = CFG_PATH.parent
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)


def cfg(path, default=None):
    cur = CFG
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def hexcol(s, a=255):
    s = s.lstrip("#")
    return skia.Color(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), a)


# ------------------------------------------------------------------ canvas & timing
W, H = cfg("video.size", [1080, 1920])
FPS = cfg("video.fps", 60)
PORTRAIT = H > W
T_INTRO = cfg("timing.intro", 2.0)
T_RACE = cfg("timing.race", 67.0)
T_HOLD = cfg("timing.hold", 1.6)
T_OUTRO = cfg("timing.outro", 4.4)
TOTAL = T_INTRO + T_RACE + T_HOLD + T_OUTRO
NF = int(round(TOTAL * FPS))

# ------------------------------------------------------------------ data
df = pd.read_csv(ROOT / cfg("data.csv", "data/values.csv"), dtype={cfg("data.code_col", "code"): str})
PCOL, CCOL = cfg("data.period_col", "period"), cfg("data.code_col", "code")
df[PCOL] = df[PCOL].astype(str)
SERIES = cfg("series")
SCOLS = [s["col"] for s in SERIES]
SORT_COL = cfg("ranking.sort_by", SCOLS[0])
DESC = cfg("ranking.order", "desc") == "desc"
periods = sorted(df[PCOL].unique())
NM = len(periods)
MONTHLY = len(periods[0]) >= 7
ENT = cfg("entities", {})
codes = sorted(df[CCOL].unique())
for c in codes:
    ENT.setdefault(c, {"name": c})
DYN = [p for p in cfg("ranking.pinned_dynamic", []) if p in ("max", "min")]
PINNED = [] if DYN else [c for c in cfg("ranking.pinned", []) if c in codes]
RACE = [c for c in codes if c not in PINNED]
TOP_N = cfg("ranking.top_n", 7)

VALS, ACTIVE = {}, {}
for c in codes:
    sub = df[df[CCOL] == c].set_index(PCOL).reindex(periods)
    ACTIVE[c] = sub[SORT_COL].notna().to_numpy()
    for s in SCOLS:
        VALS.setdefault(s, {})[c] = sub[s].to_numpy(dtype=float)


def smooth(a, w):
    """Centred moving average on the present (non-empty) span; window shrinks at both ends."""
    out = a.copy()
    if w <= 0:
        return out
    idx = np.where(~np.isnan(a))[0]
    if len(idx) == 0:
        return out
    lo, hi = idx[0], idx[-1]
    seg = a[lo:hi + 1]
    seg = pd.Series(seg).interpolate(limit_direction="both").to_numpy()
    n = len(seg)
    sm = np.empty(n)
    for i in range(n):
        k = min(w, i, n - 1 - i)
        sm[i] = seg[i - k:i + k + 1].mean()
    out[lo:hi + 1] = sm
    return out


def make_interp(a):
    idx = np.where(~np.isnan(a))[0]
    if len(idx) == 0:
        return lambda m: 0.0
    filled = a.copy()
    filled[:idx[0]] = a[idx[0]]
    filled[idx[-1] + 1:] = a[idx[-1]]
    f = PchipInterpolator(np.arange(NM), filled)
    return lambda m: float(f(m))


SW = cfg("ranking.smooth_window", 3 if MONTHLY else 0)
INTERP = {s: {c: make_interp(smooth(VALS[s][c], SW)) for c in codes} for s in SCOLS}


def is_active(c, m):
    return bool(ACTIVE[c][min(int(round(m)), NM - 1)])


# ------------------------------------------------------------------ theme
TH = cfg("theme", {})
C_BG_TOP = hexcol(TH.get("bg_top", "#181B22"))
C_BG_BOT = hexcol(TH.get("bg_bottom", "#0A0B0F"))
C_TEXT = hexcol(TH.get("text", "#F0F2F6"))
C_MUTED = hexcol(TH.get("muted", "#969DAA"))
C_DIM = hexcol(TH.get("dim", "#78808E"))
C_CARD_TOP = TH.get("card_top", "#282C36")
C_CARD_BOT = TH.get("card_bottom", "#1B1E25")
C_BAR_HI = hexcol(TH.get("bar_hi", "#F6F7FA"))
C_BAR_LO = hexcol(TH.get("bar_lo", "#C4C9D2"))
C_EXTRUDE = TH.get("bar_extrude", "#6E7480")
C_ACCENT = hexcol(TH.get("accent", SERIES[0]["color_from"]))
C_GOLD = hexcol(TH.get("leader", "#FFD36E"))
C_LOW = hexcol(TH.get("lowest", "#8FA3C7"))
GLOWS = TH.get("glows", [SERIES[0]["color_from"], SERIES[-1]["color_from"]])
DEPTH = {"none": 0, "subtle": 5, "medium": 7, "strong": 9}[TH.get("depth_3d", "subtle")]
BOKEH = TH.get("bokeh", False)
CAPS = [(hexcol(s["color_from"]), hexcol(s["color_to"])) for s in SERIES]

# ------------------------------------------------------------------ fonts
FD = cfg("fonts.dir", "/usr/share/fonts/opentype/inter/")
FFILES = {"bold": "Inter-Bold.otf", "black": "Inter-Black.otf", "semi": "Inter-SemiBold.otf",
          "med": "Inter-Medium.otf", "disp": "InterDisplay-Bold.otf"}
TF = {}
for w_, f_ in FFILES.items():
    p = Path(FD) / f_
    TF[w_] = skia.Typeface.MakeFromFile(str(p)) if p.exists() else skia.Typeface("Inter", skia.FontStyle.Bold())


def font(weight, size):
    f = skia.Font(TF[weight], size)
    f.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
    f.setSubpixel(True)
    return f


def fit_font(s, weight, size, maxw, minsize=16):
    f = font(weight, size)
    while f.measureText(s) > maxw and size > minsize:
        size -= 1
        f = font(weight, size)
    return f


# ------------------------------------------------------------------ layout (derived from canvas + row count)
U = min(W, H) / 1080.0
SAFE_L = 56 * U
HEADER_H = (400 if PORTRAIT else 250) * U
ROWS_TOP0 = HEADER_H + 86 * U
BOTTOM_LIMIT = H * (0.86 if PORTRAIT else 0.93)
SLOT_HDR = 40 * U                                   # label line above a pinned block
n_pin_rows = len(DYN) if DYN else len(PINNED)
n_blocks = (1 if "max" in DYN else 0) + (1 if ("min" in DYN or PINNED) else 0)
ROW_H = min(104 * U * (1 if PORTRAIT else 0.9),
            (BOTTOM_LIMIT - ROWS_TOP0 - n_blocks * SLOT_HDR - 30 * U) / (TOP_N + n_pin_rows))
RS = ROW_H / (104 * U)
SHOW_RANK = cfg("text.show_rank", False)
RANK_W = 46 * U * RS if SHOW_RANK else 0
FLAG_H = 50 * U * RS
FLAG_W = FLAG_H * 4 / 3
BADGE_X = SAFE_L + RANK_W
NAME_X = BADGE_X + FLAG_W + 16 * U
BAR_X0 = SAFE_L + RANK_W + (290 if PORTRAIT else 380) * U
VALUE_ROOM = (190 if any(s.get("ratio_to") for s in SERIES) else 130) * U
BAR_MAX_W = W - BAR_X0 - SAFE_L - VALUE_ROOM
NS = len(SERIES)
BAR_GAP = 8 * U * RS
BAR_H = {1: 44, 2: 30, 3: 21}[NS] * U * RS
# vertical blocks
y = ROWS_TOP0
TOP_SLOT_Y = None
if "max" in DYN:
    TOP_SLOT_Y = y + SLOT_HDR
    y = TOP_SLOT_Y + ROW_H + 6 * U
ROWS_TOP = y + (14 * U if "max" in DYN else 0)
PIN_SEP_Y = ROWS_TOP + TOP_N * ROW_H + 6 * U
PIN_TOP = PIN_SEP_Y + SLOT_HDR - 6 * U
AXIS_Y = ROWS_TOP0 if "max" not in DYN else ROWS_TOP0

# ------------------------------------------------------------------ badges
FLAGS = {}
for c in codes:
    p = ROOT / "assets" / "flags" / f"{c}.png"
    if p.exists():
        FLAGS[c] = skia.Image.open(str(p))


def initials(c):
    name = ENT[c].get("name", c)
    parts = [w for w in name.replace("-", " ").split() if w[0].isalnum()]
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else parts[0][1:2])).upper() if parts else c[:2]


def badge_color(c):
    if "color" in ENT[c]:
        return hexcol(ENT[c]["color"])
    h = sum(map(ord, c)) * 47 % 360
    r, g, b = colorsys.hls_to_rgb(h / 360, 0.52, 0.55)
    return skia.Color(int(r * 255), int(g * 255), int(b * 255))


# ------------------------------------------------------------------ helpers
def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def paint(color=None, style=None, **kw):
    p = skia.Paint(AntiAlias=True)
    if color is not None:
        p.setColor(color)
    if style == "stroke":
        p.setStyle(skia.Paint.kStroke_Style)
    for k, v in kw.items():
        getattr(p, "set" + k)(v)
    return p


def text(canvas, s, x, y, fnt, color, align="left", alpha=1.0):
    w = fnt.measureText(s)
    if align == "right":
        x -= w
    elif align == "center":
        x -= w / 2
    p = paint(color)
    p.setAlphaf(p.getAlphaf() * alpha)
    canvas.drawTextBlob(skia.TextBlob.MakeFromString(s, fnt), x, y, p)
    return w


VF = cfg("value_format", {"prefix": "", "suffix": "", "decimals": 2, "compact": False})


def fmt(v, with_unit=False):
    d = VF.get("decimals", 2)
    if VF.get("compact"):
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(v) >= div:
                s = f"{v / div:.{max(d, 1)}f}{suf}"
                break
        else:
            s = f"{v:.{d}f}"
    else:
        s = f"{v:,.{d}f}"
    return f"{VF.get('prefix', '')}{s}{VF.get('suffix', '')}" + (VF.get("unit_suffix", "") if with_unit else "")


def fmt_axis(v):
    return re.sub(r"(\d)\.0+(?=\D|$)", r"\1", fmt(v))


def month_at(t):
    if t <= T_INTRO:
        return 0.0
    if t >= T_INTRO + T_RACE:
        return NM - 1.0
    return (t - T_INTRO) / T_RACE * (NM - 1)


# ------------------------------------------------------------------ ranking simulation
SWP = cfg("ranking.swap", {})
TH_HOLD = SWP.get("threshold", 1.02)
HOLD_F = SWP.get("hold_seconds", 0.33) * FPS
COOL_F = SWP.get("cooldown_seconds", 0.5) * FPS
TH_NOW = SWP.get("instant", 1.03)
SPRING_K = SWP.get("spring_k", 0.028) * (60.0 / FPS) ** 2
SLOT_FADE = cfg("ranking.slot_fade_seconds", 0.7) * FPS
NEG = -1e18


def lead_ratio(rk, a, b):
    """>1 when b should be above a. Inactive/excluded entities carry NEG and always lose."""
    va, vb = rk[a], rk[b]
    if va <= NEG / 2 or vb <= NEG / 2:
        return 10.0 if (va <= NEG / 2 < vb) else 0.0
    eps = 1e-9
    va, vb = max(va, eps), max(vb, eps)
    return vb / va if DESC else va / vb


def precompute():
    m0 = {c: INTERP[SORT_COL][c](0.0) if is_active(c, 0) else NEG for c in RACE}
    order = sorted(RACE, key=lambda c: -m0[c] if DESC else (m0[c] if m0[c] > NEG / 2 else 1e18))
    pos, vel, last, held = {}, {}, {c: -10 ** 6 for c in RACE}, {}
    slot = {k: None for k in DYN}
    slot_prev = {k: None for k in DYN}
    slot_since = {k: -10 ** 6 for k in DYN}
    slot_hold = {k: 0 for k in DYN}
    axis, states = None, []
    for f in range(NF):
        m = month_at(f / FPS)
        act = {c: is_active(c, m) for c in codes}
        vals = {s: {c: INTERP[s][c](m) for c in codes} for s in SCOLS}
        rv = vals[SORT_COL]
        # dynamic pinned slots with hysteresis
        live = [c for c in RACE if act[c]]
        for k in DYN:
            if not live:
                continue
            best = (max if (k == "max") == DESC else min)(live, key=lambda c: rv[c])
            cur = slot[k]
            if cur is None or not act.get(cur, False):
                if cur != best:
                    slot_prev[k], slot[k], slot_since[k] = cur, best, f
                continue
            better = (rv[best] > rv[cur] * 1.02) if ((k == "max") == DESC) else (rv[best] < rv[cur] / 1.02)
            slot_hold[k] = slot_hold[k] + 1 if (best != cur and better) else 0
            if slot_hold[k] >= HOLD_F:
                slot_prev[k], slot[k], slot_since[k], slot_hold[k] = cur, best, f, 0
        excluded = {v for v in slot.values() if v}
        rk = {c: (rv[c] if act[c] and c not in excluded else NEG) for c in RACE}
        if not DESC:
            rk = {c: (v if v > NEG / 2 else NEG) for c, v in rk.items()}
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            held[(a, b)] = held.get((a, b), 0) + 1 if lead_ratio(rk, a, b) > TH_HOLD else 0
        changed = True
        while changed:
            changed = False
            for i in range(len(order) - 1):
                a, b = order[i], order[i + 1]
                lead = lead_ratio(rk, a, b)
                cool = f - max(last[a], last[b]) > COOL_F
                if (held.get((a, b), 0) >= HOLD_F and cool) or lead > TH_NOW:
                    order[i], order[i + 1] = b, a
                    last[a] = last[b] = f
                    held[(a, b)] = 0
                    changed = True
        target = {c: i for i, c in enumerate(order)}
        for c in RACE:
            if c not in pos:
                pos[c], vel[c] = float(target[c]), 0.0
            vel[c] += SPRING_K * (target[c] - pos[c]) - 2 * math.sqrt(SPRING_K) * vel[c]
            pos[c] += vel[c]
        visible = [c for c in RACE if pos[c] < TOP_N + 0.6 and act[c] and c not in excluded]
        visible += [v for v in slot.values() if v] + PINNED
        vmax = max([max(vals[s][c] for s in SCOLS) for c in visible] + [1e-9])
        want = vmax * 1.06
        axis = want if axis is None else axis + (want - axis) * 0.06
        states.append(dict(m=m, vals=vals, pos=dict(pos), axis=axis, vel=dict(vel), order=list(order),
                           act=act, excluded=set(excluded), slot=dict(slot), slot_prev=dict(slot_prev),
                           slot_since=dict(slot_since)))
    return states


def quality(states):
    prev, cross, bad = None, 0, 0
    for st in states:
        pos, order, vals = st["pos"], st["order"], st["vals"]
        live = [c for c in order if st["act"][c] and c not in st["excluded"]]
        top = [c for c in sorted(live, key=lambda c: pos[c])][:TOP_N + 1]
        if prev:
            for a in top:
                for b in top:
                    if a < b and a in prev and b in prev and (prev[a] - prev[b]) * (pos[a] - pos[b]) < 0:
                        cross += 1
        prev = {c: pos[c] for c in live}
        o = live[:TOP_N]
        rk = {c: vals[SORT_COL][c] for c in o}
        if max([lead_ratio(rk, a, b) - 1 for a, b in zip(o, o[1:])] + [0]) > 0.03:
            bad += 1
    return cross / T_RACE, 100.0 * bad / len(states)


# ------------------------------------------------------------------ drawing
rng = np.random.default_rng(3)
PARTICLES = [(rng.random() * W, rng.random() * H, (4 + rng.random() * 18) * U, 6 + rng.random() * 18,
              rng.random() * 6.28, rng.integers(0, 2)) for _ in range(34)]


def draw_background(canvas, t):
    canvas.drawRect(skia.Rect(0, 0, W, H),
                    paint(Shader=skia.GradientShader.MakeLinear([(0, 0), (0, H)], [C_BG_TOP, C_BG_BOT])))
    for (cx, cy, r, col, ph) in ((0.17 * W, 0.27 * H, 0.6 * max(W, H) * 0.55, GLOWS[0], 0.0),
                                 (0.83 * W, 0.68 * H, 0.65 * max(W, H) * 0.55, GLOWS[-1], 1.7)):
        dx, dy = math.sin(t * 0.25 + ph) * 60 * U, math.cos(t * 0.21 + ph) * 50 * U
        g = skia.GradientShader.MakeRadial((cx + dx, cy + dy), r, [hexcol(col, 30), skia.Color(0, 0, 0, 0)])
        canvas.drawRect(skia.Rect(0, 0, W, H), paint(Shader=g))
    if BOKEH:  # slow drifting soft particles
        for (x0, y0, r, sp, ph, ci) in PARTICLES:
            yy = (y0 - t * sp * U) % (H + 60 * U) - 30 * U
            xx = x0 + math.sin(t * 0.3 + ph) * 18 * U
            a = int(10 + 12 * (0.5 + 0.5 * math.sin(t * 0.7 + ph)))
            p = paint(hexcol(GLOWS[ci % len(GLOWS)], a))
            p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, r * 0.6))
            canvas.drawCircle(xx, yy, r, p)
    v = skia.GradientShader.MakeRadial((W / 2, H * 0.45), max(W, H) * 0.75,
                                       [skia.Color(0, 0, 0, 0), skia.Color(0, 0, 0, 120)], [0.55, 1.0])
    canvas.drawRect(skia.Rect(0, 0, W, H), paint(Shader=v))


def draw_badge(canvas, c, x, y, alpha, h=None):
    h = h or FLAG_H
    w = h * 4 / 3
    sh = paint(skia.Color(0, 0, 0, int(140 * alpha)))
    sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 6 * U))
    if c in FLAGS:
        r = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), 9 * U, 9 * U)
        canvas.drawRRect(r.makeOffset(0, 5 * U), sh)
        canvas.save()
        canvas.clipRRect(r, True)
        p = paint()
        p.setAlphaf(alpha)
        canvas.drawImageRect(FLAGS[c], skia.Rect.MakeXYWH(x, y, w, h),
                             skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear), p)
        gl = skia.GradientShader.MakeLinear([(0, y), (0, y + h)],
                                            [skia.Color(255, 255, 255, int(70 * alpha)), skia.Color(255, 255, 255, 0)],
                                            [0.0, 0.55])
        canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, h), paint(Shader=gl))
        canvas.restore()
        canvas.drawRRect(r, paint(skia.Color(255, 255, 255, int(55 * alpha)), style="stroke", StrokeWidth=1.2))
    else:
        cx, cy, rad = x + w / 2, y + h / 2, h * 0.62
        canvas.drawCircle(cx, cy + 4 * U, rad, sh)
        col = badge_color(c)
        r_, g_, b_ = skia.ColorGetR(col), skia.ColorGetG(col), skia.ColorGetB(col)
        light = skia.Color(min(255, r_ + 50), min(255, g_ + 50), min(255, b_ + 50))
        p = paint(Shader=skia.GradientShader.MakeLinear([(0, cy - rad), (0, cy + rad)], [light, col]))
        p.setAlphaf(alpha)
        canvas.drawCircle(cx, cy, rad, p)
        canvas.drawCircle(cx, cy, rad, paint(skia.Color(255, 255, 255, int(60 * alpha)), style="stroke", StrokeWidth=1.2))
        text(canvas, initials(c), cx, cy + h * 0.2, font("bold", h * 0.52), skia.ColorWHITE, align="center", alpha=alpha)


def draw_bar(canvas, x, y, w, h, caps, alpha, t, label, extra=None):
    w = max(w, 2)
    rad = min(8 * U, h / 2)
    sh = paint(skia.Color(0, 0, 0, int(150 * alpha)))
    sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 7 * U))
    canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x + 3 * U, y + 8 * U, w, h), rad, rad), sh)
    if DEPTH:  # extruded bottom face + darker side for a little 3D
        canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x + 1.5 * U, y + DEPTH * U, w, h), rad, rad),
                         paint(hexcol(C_EXTRUDE, int(255 * alpha))))
        side = skia.GradientShader.MakeLinear([(0, y + h - 2 * U), (0, y + h + DEPTH * U)],
                                              [caps[1], hexcol(C_EXTRUDE)])
        sp = paint(Shader=side)
        sp.setAlphaf(alpha * 0.85)
        canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x + w - 9 * U, y + h - 3 * U, 9 * U, DEPTH * U + 3 * U),
                                               3 * U, 3 * U), sp)
    face = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), rad, rad)
    fp = paint(Shader=skia.GradientShader.MakeLinear([(0, y), (0, y + h)], [C_BAR_HI, C_BAR_LO]))
    fp.setAlphaf(alpha)
    canvas.drawRRect(face, fp)
    canvas.save()
    canvas.clipRRect(face, True)
    cap_w = min(w, 120 * U)
    cp = paint(Shader=skia.GradientShader.MakeLinear([(x + w - cap_w, 0), (x + w, 0)], [skia.Color(0, 0, 0, 0), caps[0]]))
    cp.setAlphaf(alpha * 0.95)
    canvas.drawRect(skia.Rect.MakeXYWH(x + w - cap_w, y, cap_w, h), cp)
    tip = paint(caps[1])
    tip.setAlphaf(alpha)
    canvas.drawRect(skia.Rect.MakeXYWH(x + w - 7 * U, y, 7 * U, h), tip)
    sx = x + ((t * 220 * U) % (w + 300 * U)) - 150 * U
    canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, h), paint(Shader=skia.GradientShader.MakeLinear(
        [(sx - 60 * U, 0), (sx + 60 * U, 0)],
        [skia.Color(255, 255, 255, 0), skia.Color(255, 255, 255, int(70 * alpha)), skia.Color(255, 255, 255, 0)])))
    canvas.drawRect(skia.Rect.MakeXYWH(x + 6 * U, y + 2 * U, max(w - 12 * U, 0), 2 * U),
                    paint(skia.Color(255, 255, 255, int(170 * alpha))))
    canvas.restore()
    gl = paint(caps[0])
    gl.setAlphaf(0.55 * alpha)
    gl.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 9 * U))
    canvas.drawCircle(x + w - 3 * U, y + h / 2, h * 0.42, gl)
    fsz = max(18 * U, min(25 * U * RS, h * 0.9 + 4 * U))
    tw = text(canvas, label, x + w + 14 * U, y + h / 2 + fsz * 0.36, font("bold", fsz), C_TEXT, alpha=alpha)
    if extra:
        text(canvas, extra, x + w + 14 * U + tw + 9 * U, y + h / 2 + fsz * 0.36, font("semi", fsz * 0.8), caps[0],
             alpha=alpha * 0.9)


def draw_row(canvas, c, y, vals, axis, alpha, t, grow, rank=None, lift=0.0, ring=None):
    if alpha <= 0.01:
        return
    canvas.save()
    if lift > 0.001:  # a moving row lifts towards the viewer: slight scale + deeper shadow
        s = 1.0 + 0.035 * lift
        cx, cy = W / 2, y + ROW_H / 2
        canvas.translate(cx, cy)
        canvas.scale(s, s)
        canvas.translate(-cx, -cy)
    rect = skia.Rect.MakeXYWH(SAFE_L - 16 * U, y - 2 * U, W - 2 * SAFE_L + 32 * U, ROW_H - 10 * U)
    card = skia.RRect.MakeRectXY(rect, 18 * U, 18 * U)
    sh = paint(skia.Color(0, 0, 0, int((70 + 110 * lift) * alpha)))
    sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, (10 + 14 * lift) * U))
    canvas.drawRRect(card.makeOffset(0, (6 + 10 * lift) * U), sh)
    canvas.drawRRect(card, paint(Shader=skia.GradientShader.MakeLinear(
        [(0, y), (0, y + ROW_H)], [hexcol(C_CARD_TOP, int(240 * alpha)), hexcol(C_CARD_BOT, int(240 * alpha))])))
    if ring is not None:  # leader / lowest outline
        rp = paint(ring, style="stroke", StrokeWidth=2.0 * U)
        rp.setAlphaf(0.85 * alpha)
        canvas.drawRRect(card, rp)
        gp = paint(ring, style="stroke", StrokeWidth=6 * U)
        gp.setAlphaf(0.25 * alpha)
        gp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 8 * U))
        canvas.drawRRect(card, gp)
    else:
        canvas.drawRRect(card, paint(skia.Color(255, 255, 255, int(16 * alpha)), style="stroke", StrokeWidth=1.2))
    mid = y + (ROW_H - 10 * U) / 2
    if rank is not None and SHOW_RANK:
        text(canvas, str(rank), SAFE_L + RANK_W / 2 - 4 * U, mid + 11 * U * RS, font("black", 30 * U * RS),
             C_MUTED if ring is None else ring, align="center", alpha=alpha)
    draw_badge(canvas, c, BADGE_X, mid - FLAG_H / 2, alpha)
    name = ENT[c].get("name", c)
    sub = ENT[c].get("sub", c if c != name else "")
    f = fit_font(name, "semi", 29 * U * RS, BAR_X0 - NAME_X - 14 * U)
    text(canvas, name, NAME_X, mid + (2 if sub else 10) * U * RS, f, C_TEXT, alpha=alpha)
    if sub:
        text(canvas, sub, NAME_X, mid + 28 * U * RS, font("med", 19 * U * RS), C_MUTED, alpha=alpha * 0.8)
    scale = BAR_MAX_W / axis
    block = NS * BAR_H + (NS - 1) * BAR_GAP
    by = mid - block / 2 - 2 * U
    for i, s in enumerate(SERIES):
        v = vals[s["col"]][c]
        extra = None
        if s.get("ratio_to") and vals[s["ratio_to"]][c] > 0:
            extra = f"{100 * v / vals[s['ratio_to']][c]:.0f}%"
        draw_bar(canvas, BAR_X0, by + i * (BAR_H + BAR_GAP), max(v, 0) * scale * grow, BAR_H, CAPS[i], alpha,
                 t + (sum(map(ord, c)) % 7) + 1.3 * i, fmt(v), extra)
    canvas.restore()


MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
SHOW_MONTH = cfg("text.show_month", True)


def period_label(m):
    p = periods[min(int(round(m)), NM - 1)]
    return p[:4], (MONTHS[int(p[5:7]) - 1] if MONTHLY and SHOW_MONTH else "")


def draw_header(canvas, m, k, t):
    a = ease_out(k)
    dy = (1 - a) * -30 * U
    top = (262 if PORTRAIT else 120) * U
    tf = fit_font(cfg("text.title", ""), "black", 58 * U, W * (0.6 if PORTRAIT else 0.62) - SAFE_L)
    text(canvas, cfg("text.title", ""), SAFE_L, top + dy, tf, C_TEXT, alpha=a)
    sf = fit_font(cfg("text.subtitle", ""), "med", 27 * U, W * (0.62 if PORTRAIT else 0.62) - SAFE_L)
    text(canvas, cfg("text.subtitle", ""), SAFE_L, top + 44 * U + dy, sf, C_MUTED, alpha=a)
    lx, ly = SAFE_L, top + 76 * U
    if NS > 1 or cfg("text.show_legend", False):
        for s, caps in zip(SERIES, CAPS):
            canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(lx, ly + 4 * U, 26 * U, 16 * U), 5 * U, 5 * U),
                             paint(Shader=skia.GradientShader.MakeLinear([(lx, 0), (lx + 26 * U, 0)], list(caps)),
                                   Alphaf=a))
            w = text(canvas, s.get("label", s["col"]), lx + 36 * U, ly + 20 * U, font("semi", 23 * U), C_TEXT, alpha=a)
            lx += 36 * U + w + 34 * U
    year, mon = period_label(m)
    # year pulse: briefly scale up when the year changes
    pi = int(round(m))
    first_idx = next((i for i in range(pi, -1, -1) if periods[i][:4] != year), -1) + 1
    t_change = T_INTRO + first_idx / max(NM - 1, 1) * T_RACE
    pulse = 1.0 + 0.07 * max(0.0, 1.0 - (t - t_change) / 0.35) if t > T_INTRO + 0.1 else 1.0
    yf = font("disp", 132 * U * pulse)
    yw = yf.measureText(year)
    ybase = top + 130 * U
    gp = paint(hexcol(cfg("theme.year_glow", "#FFFFFF"), int(30 + 60 * (pulse - 1) / 0.07)))
    gp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 22 * U))
    canvas.drawTextBlob(skia.TextBlob.MakeFromString(year, yf), W - SAFE_L - yw, ybase, gp)
    text(canvas, year, W - SAFE_L, ybase, yf, C_TEXT, align="right", alpha=a)
    if mon:
        text(canvas, mon, W - SAFE_L - 4 * U, top, font("bold", 30 * U), C_MUTED, align="right", alpha=a)


def nice_step(axis):
    raw = axis / 6
    mag = 10 ** math.floor(math.log10(raw))
    for k in (1, 2, 2.5, 5, 10):
        if raw <= k * mag:
            return k * mag
    return 10 * mag


def draw_axis(canvas, axis, alpha):
    step = nice_step(axis)
    v = step
    f = font("med", 19 * U)
    while v < axis * 0.98:
        x = BAR_X0 + v / axis * BAR_MAX_W
        canvas.drawRect(skia.Rect.MakeXYWH(x - 0.6, AXIS_Y - 12 * U, 1.2, 8 * U),
                        paint(skia.Color(255, 255, 255, int(60 * alpha))))
        text(canvas, fmt_axis(v), x, AXIS_Y - 16 * U, f, C_DIM, align="center", alpha=alpha)
        v += step


EVENTS = [e for e in cfg("events", []) if e[0] in periods]
EVENT_DUR = cfg("timing.event_seconds", 2.8)


def draw_event(canvas, t):
    starts = [T_INTRO + periods.index(e[0]) / (NM - 1) * T_RACE for e in EVENTS]
    for i, (lbl, msg) in enumerate(EVENTS):
        dur = min(EVENT_DUR, (starts[i + 1] - starts[i] - 0.05) if i + 1 < len(EVENTS) else EVENT_DUR)
        if dur <= 0.3:
            continue
        k = (t - starts[i]) / dur
        if 0 <= k <= 1:
            a = ease_in_out(min(1.0, k / 0.12, (1 - k) / 0.15))
            f = fit_font(msg, "semi", 25 * U, W - 2 * SAFE_L - 60 * U)
            w = f.measureText(msg) + 44 * U
            x, y = SAFE_L, ((392 if PORTRAIT else 250) * U) - (1 - a) * 12 * U
            r = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, 46 * U), 23 * U, 23 * U)
            canvas.drawRRect(r, paint(skia.Color(255, 255, 255, int(28 * a))))
            canvas.drawRRect(r, paint(C_ACCENT, style="stroke", StrokeWidth=1.6, Alphaf=0.67 * a))
            canvas.drawCircle(x + 20 * U, y + 23 * U, 5 * U, paint(C_ACCENT, Alphaf=a))
            text(canvas, msg, x + 34 * U, y + 32 * U, f, C_TEXT, alpha=a)


def draw_block_label(canvas, y, label, color, alpha, icon=None):
    f = font("bold", 19 * U)
    x = SAFE_L
    if icon == "crown":
        p = skia.Path()
        cx, cy, s = x + 13 * U, y - 7 * U, 12 * U
        p.moveTo(cx - s, cy + s * 0.55)
        p.lineTo(cx - s, cy - s * 0.35)
        p.lineTo(cx - s * 0.45, cy + s * 0.1)
        p.lineTo(cx, cy - s * 0.6)
        p.lineTo(cx + s * 0.45, cy + s * 0.1)
        p.lineTo(cx + s, cy - s * 0.35)
        p.lineTo(cx + s, cy + s * 0.55)
        p.close()
        canvas.drawPath(p, paint(color, Alphaf=alpha))
        x += 32 * U
    elif icon == "down":
        p = skia.Path()
        cx, cy, s = x + 10 * U, y - 7 * U, 9 * U
        p.moveTo(cx - s, cy - s * 0.5)
        p.lineTo(cx + s, cy - s * 0.5)
        p.lineTo(cx, cy + s * 0.7)
        p.close()
        canvas.drawPath(p, paint(color, Alphaf=alpha))
        x += 28 * U
    w = text(canvas, label, x, y, f, color, alpha=alpha)
    canvas.drawRect(skia.Rect.MakeXYWH(x + w + 16 * U, y - 7 * U, W - SAFE_L - (x + w + 16 * U), 1.5),
                    paint(skia.Color(255, 255, 255, int(40 * alpha))))


def draw_slot(canvas, st, key, y, vals, axis, rows_a, t, grow, f, rank=None):
    cur, prev, since = st["slot"].get(key), st["slot_prev"].get(key), st["slot_since"].get(key, -10 ** 6)
    k = min(1.0, (f - since) / SLOT_FADE) if since > 0 else 1.0
    k = ease_in_out(k)
    ring = C_GOLD if key == "max" else C_LOW
    if prev and k < 1.0:
        draw_row(canvas, prev, y + 18 * U * k, vals, axis, rows_a * (1 - k), t, grow, rank=rank, ring=ring)
    if cur:
        draw_row(canvas, cur, y - 18 * U * (1 - k), vals, axis, rows_a * k, t, grow, rank=rank, ring=ring)


def draw_footer(canvas, alpha):
    s = cfg("text.footer", "")
    if s:
        n_bottom = (1 if "min" in DYN else 0) if DYN else len(PINNED)
        y = (PIN_TOP + n_bottom * ROW_H if n_bottom else ROWS_TOP + TOP_N * ROW_H) + 14 * U
        text(canvas, s, SAFE_L, y, fit_font(s, "med", 19 * U, W - 2 * SAFE_L), C_DIM, alpha=alpha)


def draw_outro(canvas, k, vals, act):
    if k <= 0:
        return
    a = ease_in_out(min(1.0, k / 0.18))
    canvas.drawRect(skia.Rect(0, 0, W, H), paint(skia.Color(6, 7, 10, int(170 * a))))
    sources = cfg("sources", [])
    ch = (180 + 3 * 62 + (52 + 40 * len(sources) if sources else 0) + (62 if cfg('text.cta') else 0) + 40) * U
    cw = min(W - 2 * SAFE_L + 16 * U, 1000 * U)
    card = skia.Rect.MakeXYWH((W - cw) / 2, (H - ch) / 2 - (0.05 * H if PORTRAIT else 0) + (1 - a) * 60 * U, cw, ch)
    rr = skia.RRect.MakeRectXY(card, 28 * U, 28 * U)
    sh = paint(skia.Color(0, 0, 0, int(160 * a)))
    sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 24 * U))
    canvas.drawRRect(rr.makeOffset(0, 16 * U), sh)
    canvas.drawRRect(rr, paint(Shader=skia.GradientShader.MakeLinear(
        [(0, card.top()), (0, card.bottom())], [hexcol(C_CARD_TOP, int(245 * a)), hexcol(C_CARD_BOT, int(245 * a))])))
    canvas.drawRRect(rr, paint(skia.Color(255, 255, 255, int(40 * a)), style="stroke", StrokeWidth=1.5))
    x0, y = card.left() + 44 * U, card.top() + 84 * U
    title = cfg("text.outro_title", "Final ranking")
    text(canvas, title, x0, y, fit_font(title, "bold", 34 * U, card.width() - 88 * U), C_TEXT, alpha=a)
    y += 26 * U
    rk = vals[SORT_COL]
    live = [c for c in codes if act[c]]
    final = sorted(live, key=lambda c: -rk[c] if DESC else rk[c])[:3]
    for i, c in enumerate(final):
        y += 62 * U
        draw_badge(canvas, c, x0, y - 40 * U, a, h=50 * U)
        text(canvas, f"{i + 1}. {ENT[c].get('name', c)}", x0 + 66 * U + 18 * U, y, font("semi", 30 * U), C_TEXT, alpha=a)
        text(canvas, fmt(rk[c], with_unit=True), card.right() - 44 * U, y, font("bold", 30 * U),
             CAPS[SCOLS.index(SORT_COL)][0], align="right", alpha=a)
    y += 70 * U
    canvas.drawRect(skia.Rect.MakeXYWH(x0, y, card.width() - 88 * U, 1.5), paint(skia.Color(255, 255, 255, int(40 * a))))
    if sources:
        y += 52 * U
        text(canvas, cfg("text.sources_label", "DATA SOURCES"), x0, y, font("bold", 22 * U), C_MUTED, alpha=a)
        for s in sources:
            y += 40 * U
            text(canvas, s, x0, y, fit_font(s, "med", 23 * U, card.width() - 88 * U), skia.Color(205, 210, 220), alpha=a)
    cta = cfg("text.cta", "")
    if cta:
        y += 62 * U
        text(canvas, cta, x0, y, fit_font(cta, "bold", 30 * U, card.width() - 88 * U), CAPS[-1][0], alpha=a)


def render_frame(canvas, f, st):
    t = f / FPS
    m, vals, pos, axis, vel = st["m"], st["vals"], st["pos"], st["axis"], st["vel"]
    act, excl = st["act"], st["excluded"]
    canvas.clear(skia.ColorBLACK)
    draw_background(canvas, t)
    grow = ease_out((t - 0.4) / 1.4)
    rows_a = ease_out((t - 0.25) / 0.8)
    draw_header(canvas, m, t / 0.9, t)
    draw_axis(canvas, axis, rows_a)
    draw_event(canvas, t)
    rank_offset = 1 if "max" in DYN else 0
    if "max" in DYN:
        draw_block_label(canvas, TOP_SLOT_Y - 12 * U, cfg("text.max_label", "HIGHEST"), C_GOLD, rows_a, icon="crown")
        draw_slot(canvas, st, "max", TOP_SLOT_Y, vals, axis, rows_a, t, grow, f, rank=1)
    canvas.save()
    canvas.clipRect(skia.Rect(0, ROWS_TOP - 30 * U, W, PIN_SEP_Y - 2))
    live = [c for c in RACE if act[c] and c not in excl]
    by_pos = sorted(live, key=lambda c: pos[c])
    ranks = {c: i + 1 + rank_offset for i, c in enumerate(by_pos)}
    for c in sorted(live, key=lambda c: (vel[c] <= 0.002, -pos[c])):  # falling rows under rising ones
        p = pos[c]
        if p > TOP_N + 0.5:
            continue
        alpha = min(1.0, max(0.0, TOP_N + 0.15 - p)) * rows_a
        if vel[c] > 0.002:
            alpha *= 1.0 - min(0.4, vel[c] * 9)
        lift = min(1.0, abs(vel[c]) * 14) if vel[c] < -0.002 else 0.0
        draw_row(canvas, c, ROWS_TOP + p * ROW_H, vals, axis, alpha, t, grow, rank=ranks[c], lift=lift)
    canvas.restore()
    if "min" in DYN:
        draw_block_label(canvas, PIN_SEP_Y + 18 * U, cfg("text.min_label", "LOWEST"), C_LOW, rows_a, icon="down")
        n_live = sum(1 for c in RACE if act[c])
        draw_slot(canvas, st, "min", PIN_TOP, vals, axis, rows_a, t, grow, f, rank=n_live)
    elif PINNED:
        draw_block_label(canvas, PIN_SEP_Y + 18 * U, cfg("text.pinned_label", "FOR COMPARISON"), C_MUTED, rows_a)
        for i, c in enumerate(PINNED):
            if act[c]:
                draw_row(canvas, c, PIN_TOP + i * ROW_H, vals, axis, rows_a, t, grow)
    draw_footer(canvas, rows_a)
    t_out = T_INTRO + T_RACE + T_HOLD
    if t >= t_out:
        draw_outro(canvas, (t - t_out) / T_OUTRO, vals, act)


def main():
    print(f"{W}x{H}@{FPS}  {TOTAL:.1f}s  {NF} frames  {NM} periods  {len(RACE)} racing + {len(PINNED)} pinned"
          + (f" + dynamic {DYN}" if DYN else ""), flush=True)
    states = precompute()
    if ARGS.check:
        cps, bad = quality(states)
        print(f"swaps per second in visible top: {cps:.2f} (target <= 2.5)")
        print(f"frames with settled order wrong by >3%: {bad:.1f}% (target 0)")
        ch = [(round(f / FPS, 1), k, st["slot"][k]) for f, st in enumerate(states) for k in DYN
              if st["slot_since"][k] == f]
        if ch:
            print("pinned slot changes (s, slot, entity):", ch)
        return
    surface = skia.Surface(W, H)
    if ARGS.preview is not None:
        (BUILD / "frames").mkdir(exist_ok=True)
        ev = [int((T_INTRO + periods.index(e[0]) / (NM - 1) * T_RACE + 0.8) * FPS) for e in EVENTS]
        frames = ARGS.preview or sorted(set([int(1.5 * FPS), int(NF * 0.3), int(NF * 0.6)] + ev +
                                            [int((T_INTRO + T_RACE) * FPS) - 2, NF - 10]))
        for f in frames:
            with surface as canvas:
                render_frame(canvas, f, states[min(f, NF - 1)])
            surface.makeImageSnapshot().save(str(BUILD / "frames" / f"f{f:05d}.png"), skia.kPNG)
        print("previews:", ", ".join(f"build/frames/f{f:05d}.png" for f in frames))
        return
    out = ARGS.out or str(BUILD / "race_silent.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    end = NF if ARGS.end is None else min(ARGS.end, NF)
    for f in range(ARGS.start, end):
        with surface as canvas:
            render_frame(canvas, f, states[f])
        ff.stdin.write(surface.makeImageSnapshot().tobytes())
        if f % 600 == 0:
            print("frame", f, "/", NF, flush=True)
    ff.stdin.close()
    ff.wait()
    print("done", out)


if __name__ == "__main__":
    main()
