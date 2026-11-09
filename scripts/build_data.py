"""Build data/values.csv: monthly gross minimum wage vs gross average wage, EUR/month, Jul 2000 - Jul 2025.

Sources (data/raw/eurostat_wages.txt, Eurostat API, downloaded 2026-10-31):
  MW  = earn_mw_cur  : statutory minimum wage, EUR/month gross, values on 1 Jan (S1) and 1 Jul (S2)
  FTE = nama_10_fte  : average full-time adjusted salary per employee, EUR/year gross (1999-2024)
  AW  = earn_nt_net  : gross earnings of the average worker (GRS, P1_NCH_AW100), EUR/year (2000-2025)

Average wage (EUR/month) = FTE / 12, placed at mid-year (July).
  * 2025 = FTE 2024 x AW growth 2024->2025 (same methodology both years).
  * Netherlands has no nama_10_fte series: AW is used 2000-2023; 2024 = AW 2023 x euro-area FTE growth
    2023->2024 (AW NL shows a methodological break in 2024); 2025 = that x AW NL growth 2024->2025.
  * Missing early FTE years (BG 2000, PL 2000) are backcast with the next year's growth rate (estimate).
Minimum wage between the 1 Jan / 1 Jul points is joined with monotone PCHIP; a country is in the race only
from its first minimum-wage observation (DE 2015, CY 2023, HR 2008, IE Jul 2000).

Run: python3 scripts/build_data.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "eurostat_wages.txt"
START, END = (2000, 7), (2025, 7)
N = (END[0] - START[0]) * 12 + END[1] - START[1] + 1


def mi(y, m):
    return (y - START[0]) * 12 + m - START[1]


def lab(i):
    y, m = divmod(i + START[1] - 1, 12)
    return f"{START[0] + y}-{m + 1:02d}"


data = {"MW": {}, "FTE": {}, "AW": {}}
for line in RAW.read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    kind, geo, rest = line.split(" ", 2)
    d = {}
    for p in rest.split(","):
        k, v = p.split(":")
        d[k] = float(v)
    data[kind][geo] = d

COUNTRIES = ["BE", "BG", "CZ", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "CY", "LV", "LT", "LU", "HU", "MT",
             "NL", "PL", "PT", "RO", "SI", "SK"]
notes = {}

# ---------------------------------------------------------------- minimum wage (monthly, NaN = no minimum wage yet)
mw = {}
for c in COUNTRIES:
    pts = {}
    for k, v in data["MW"][c].items():
        y, s = k.split("-S")
        pts[mi(int(y), 1 if s == "1" else 7)] = v
    xs = sorted(pts)
    f = PchipInterpolator(xs, [pts[x] for x in xs], extrapolate=False)
    arr = f(np.arange(N, dtype=float))
    first = xs[0] if xs[0] > 0 else 0
    arr[:max(first, 0)] = np.nan
    if xs[0] <= 0:  # starts before the window: value at window start is exact from the curve
        arr[0] = f(0.0)
    mw[c] = arr

# ---------------------------------------------------------------- average wage (annual -> monthly)
def annual_avg(c):
    fte = {int(k): v for k, v in data["FTE"].get(c, {}).items()}
    aw = {int(k): v for k, v in data["AW"].get(c, {}).items()}
    note = "nama_10_fte"
    if c == "NL":
        ann = {y: v for y, v in aw.items() if y <= 2023}
        ea = {int(k): v for k, v in data["FTE"]["EA20"].items()}
        ann[2024] = ann[2023] * ea[2024] / ea[2023]
        ann[2025] = ann[2024] * aw[2025] / aw[2024]
        note = "earn_nt_net AW 2000-2023; 2024 via euro-area FTE growth (estimate); 2025 via AW growth"
    else:
        ann = dict(fte)
        ann[2025] = ann[2024] * aw[2025] / aw[2024]
        note += "; 2025 via earn_nt_net AW growth"
    for y in (2000,):
        if y not in ann and y + 1 in ann and y + 2 in ann:
            ann[y] = ann[y + 1] / (ann[y + 2] / ann[y + 1])
            note += f"; {y} backcast (estimate)"
    return ann, note


avg = {}
for c in COUNTRIES:
    ann, notes[c] = annual_avg(c)
    xs = [mi(y, 7) for y in sorted(ann) if START[0] - 1 <= y <= END[0]]
    ys = [ann[y] / 12.0 for y in sorted(ann) if START[0] - 1 <= y <= END[0]]
    f = PchipInterpolator(xs, ys, extrapolate=True)
    a = f(np.arange(N, dtype=float))
    a[np.isnan(mw[c])] = np.nan
    avg[c] = a

# ---------------------------------------------------------------- write
rows = ["period,code,minimum,average"]
for i in range(N):
    for c in COUNTRIES:
        m_, a_ = mw[c][i], avg[c][i]
        rows.append(f"{lab(i)},{c}," + ("" if np.isnan(m_) else f"{m_:.2f}") + "," + ("" if np.isnan(a_) else f"{a_:.2f}"))
(ROOT / "data" / "values.csv").write_text("\n".join(rows) + "\n")
(ROOT / "data" / "series_notes.json").write_text(json.dumps(notes, indent=2))
last = {c: (round(mw[c][-1]), round(avg[c][-1]), round(100 * mw[c][-1] / avg[c][-1])) for c in COUNTRIES}
print("periods", N, lab(0), "->", lab(N - 1))
print("Jul 2025 (min, avg, %):", sorted(last.items(), key=lambda kv: -kv[1][0])[:5], "...",
      sorted(last.items(), key=lambda kv: kv[1][0])[:2])
