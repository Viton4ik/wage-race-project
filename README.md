# Minimum vs average wage, EU 2000-2025 (Shorts / Reels)

1080x1920, 60 fps, 73 s. 22 EU countries with a statutory minimum wage; top 8 race by minimum wage,
the highest pinned on top (gold) and the lowest pinned at the bottom, both cross-fading when they change.

Rebuild everything:
```
pip install numpy pandas scipy skia-python cairosvg   # + ffmpeg, npm, Inter font
python3 scripts/build_data.py
engine/make_video.sh config.json
```
Change colours, texts, timing, music: `config.json`. Data and method: `data/SOURCES.md`.
