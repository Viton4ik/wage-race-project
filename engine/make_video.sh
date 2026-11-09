#!/usr/bin/env bash
# One command: flags -> quality check -> music -> parallel render -> mux (loudness-normalised).
# Usage: engine/make_video.sh [config.json]
set -euo pipefail
CONFIG=${1:-config.json}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOTDIR="$(cd "$(dirname "$CONFIG")" && pwd)"
OUT=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print(c.get('output','output/video.mp4'))" "$CONFIG")
LUFS=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print(c.get('music',{}).get('lufs',-14))" "$CONFIG")
MODE=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print(c.get('music',{}).get('mode','synth'))" "$CONFIG")
python3 "$HERE/make_flags.py" --config "$CONFIG"
python3 "$HERE/render.py" --config "$CONFIG" --check 2>/dev/null
"$HERE/render_parallel.sh" "$CONFIG"
mkdir -p "$ROOTDIR/$(dirname "$OUT")"
if [ "$MODE" = "none" ]; then
  cp "$ROOTDIR/build/race_silent.mp4" "$ROOTDIR/$OUT"
else
  if [ "$MODE" = "file" ]; then
    MFILE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['music']['file'])" "$CONFIG")
    DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$ROOTDIR/build/race_silent.mp4")
    ffmpeg -y -loglevel error -i "$ROOTDIR/$MFILE" -t "$DUR" -af "afade=t=out:st=$(python3 -c "print(max(0,$DUR-3))"):d=3" "$ROOTDIR/build/music.wav"
  else
    python3 "$HERE/music.py" --config "$CONFIG"
  fi
  ffmpeg -y -loglevel error -i "$ROOTDIR/build/race_silent.mp4" -i "$ROOTDIR/build/music.wav" \
    -filter_complex "[1:a]loudnorm=I=$LUFS:TP=-1.5:LRA=9[a]" -map 0:v -map "[a]" \
    -c:v copy -c:a aac -b:a 192k -ar 48000 -shortest -movflags +faststart "$ROOTDIR/$OUT"
fi
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate -of compact "$ROOTDIR/$OUT"
echo "done: $OUT"
