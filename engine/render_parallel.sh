#!/usr/bin/env bash
# Render in N parallel chunks (default: CPU cores) and join without re-encoding.
# Usage: engine/render_parallel.sh [config.json] [N]
set -euo pipefail
CONFIG=${1:-config.json}
N=${2:-$(nproc)}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOTDIR="$(cd "$(dirname "$CONFIG")" && pwd)"
TOTAL=$(python3 "$HERE/render.py" --config "$CONFIG" --check 2>/dev/null | sed -nE '1s/.* ([0-9]+) frames.*/\1/p')
CH=$(( (TOTAL + N - 1) / N ))
mkdir -p "$ROOTDIR/build"
: > "$ROOTDIR/build/chunks.txt"
for i in $(seq 0 $((N-1))); do
  python3 "$HERE/render.py" --config "$CONFIG" --start $((i*CH)) --end $(((i+1)*CH)) \
    --out "$ROOTDIR/build/chunk_$i.mp4" > "$ROOTDIR/build/render_$i.log" 2>&1 &
  echo "file 'chunk_$i.mp4'" >> "$ROOTDIR/build/chunks.txt"
done
wait
ffmpeg -y -loglevel error -f concat -safe 0 -i "$ROOTDIR/build/chunks.txt" -c copy "$ROOTDIR/build/race_silent.mp4"
rm -f "$ROOTDIR"/build/chunk_*.mp4 "$ROOTDIR"/build/render_*.log "$ROOTDIR/build/chunks.txt"
echo "joined build/race_silent.mp4"
