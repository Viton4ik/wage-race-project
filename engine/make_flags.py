"""Render country flags (flag-icons, MIT licence) to assets/flags/<CODE>.png for every entity with "flag".

config.json -> "entities": {"DK": {"name": "Denmark", "flag": "dk"}, ...}   (flag = ISO 3166-1 alpha-2, lowercase;
also "gb-eng", "gb-sct", "eu", "un" etc. exist in flag-icons). Entities without "flag" get an initials badge.

Run: python3 engine/make_flags.py --config config.json      (needs: npm, cairosvg)
"""
import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import cairosvg

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.environ.get("CONFIG", "config.json"))
cfg_path = Path(ap.parse_args().config).resolve()
root = cfg_path.parent
ents = json.loads(cfg_path.read_text(encoding="utf-8")).get("entities", {})
want = {c: e["flag"].lower() for c, e in ents.items() if e.get("flag")}
out = root / "assets" / "flags"
out.mkdir(parents=True, exist_ok=True)
if not want:
    print("no flags requested")
    raise SystemExit
tmp = Path(tempfile.mkdtemp())
subprocess.run(["npm", "pack", "flag-icons", "--silent"], cwd=tmp, check=True, capture_output=True)
tgz = next(tmp.glob("flag-icons-*.tgz"))
with tarfile.open(tgz) as t:
    t.extractall(tmp, filter="data")
svgdir = tmp / "package" / "flags" / "4x3"
missing = []
for code, iso in want.items():
    svg = svgdir / f"{iso}.svg"
    if not svg.exists():
        missing.append(f"{code}:{iso}")
        continue
    cairosvg.svg2png(url=str(svg), write_to=str(out / f"{code}.png"), output_width=160, output_height=120)
shutil.copy(tmp / "package" / "LICENSE", out / "LICENSE-flag-icons.txt")
shutil.rmtree(tmp)
print(f"flags: {len(want) - len(missing)} rendered" + (f"; missing (will use initials): {missing}" if missing else ""))
