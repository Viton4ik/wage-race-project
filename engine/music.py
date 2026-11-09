"""Synthesize an original, royalty-free background track (no samples, no third-party audio).

Style: modern cinematic electro (default 100 BPM, A minor Am-F-C-G), intro swell, driving groove during the
race, soft impact when the outro card appears, fade-out. Length and timing come from config.json.

Run:  python3 engine/music.py --config config.json  -> build/music.wav
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, fftconvolve

_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default=os.environ.get("CONFIG", "config.json"))
_cfg_path = Path(_ap.parse_known_args()[0].config).resolve()
CFG = json.loads(_cfg_path.read_text(encoding="utf-8"))
ROOT = _cfg_path.parent
_t = CFG.get("timing", {})
_m = CFG.get("music", {})
SR = 44100
BPM = _m.get("bpm", 100)
BEAT = 60 / BPM
DUR = _t.get("intro", 2.0) + _t.get("race", 67.0) + _t.get("hold", 1.6) + _t.get("outro", 4.4)
N = int(SR * DUR)
rng = np.random.default_rng(_m.get("seed", 7))

L = np.zeros(N)
R = np.zeros(N)


def note_hz(n):  # MIDI -> Hz
    return 440.0 * 2 ** ((n - 69) / 12)


def env_adsr(n, a=0.01, d=0.1, s=0.7, r=0.2, sr=SR):
    a_n, d_n, r_n = int(a * sr), int(d * sr), int(r * sr)
    s_n = max(n - a_n - d_n - r_n, 0)
    e = np.concatenate([np.linspace(0, 1, a_n, endpoint=False), np.linspace(1, s, d_n, endpoint=False),
                        np.full(s_n, s), np.linspace(s, 0, r_n)])
    return e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)))


def add(sig, start_s, pan=0.0, gain=1.0):
    i = int(start_s * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * lg * 1.414
    R[i:i + len(sig)] += sig * rg * 1.414


def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def saw(f, n, detune=0.0):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for d in (-detune, 0, detune):
        ph = (t * f * (1 + d)) % 1.0
        out += 2 * ph - 1
    return out / 3


# ------------------------------------------------------------------ arrangement
# MIDI voicings; default Am F C G. Override in config: "music": {"chords": [[...],...], "bass": [...]}
CHORDS = _m.get("chords", [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]])
BASS = _m.get("bass", [45, 41, 48, 43])
_shift = _m.get("transpose", 0)
CHORDS = [[n + _shift for n in ch] for ch in CHORDS]
BASS = [n + _shift for n in BASS]
BAR = 4 * BEAT
n_bars = int(DUR / BAR) + 1
OUTRO_T = _t.get("intro", 2.0) + _t.get("race", 67.0) + _t.get("hold", 1.6)  # outro card starts
INTRO_BARS, OUTRO_START = 1, round(OUTRO_T / BAR)  # groove runs until the outro card

# pads (whole track), slow attack, filtered, wide
for b in range(n_bars):
    ch = CHORDS[b % len(CHORDS)]
    n = int(BAR * SR * 1.05)
    for k, m in enumerate(ch + [ch[0] + 12]):
        s = saw(note_hz(m), n, detune=0.004)
        s = lp(s, 1400 if b > INTRO_BARS else 700)
        s *= env_adsr(n, a=0.6, d=0.4, s=0.8, r=0.6)
        add(s, b * BAR, pan=(-0.6 + 0.4 * k), gain=0.055)

# sub bass on 8ths with soft envelope
for b in range(INTRO_BARS, int(OUTRO_START)):
    root = BASS[b % len(BASS)]
    for e in range(8):
        n = int(BEAT / 2 * SR)
        t = np.arange(n) / SR
        f = note_hz(root if e % 4 != 3 else root + 12)
        s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)
        s = lp(s, 380) * env_adsr(n, a=0.005, d=0.12, s=0.5, r=0.05)
        add(s, b * BAR + e * BEAT / 2, gain=0.20)

# kick on every beat (+ sidechain envelope)
duck = np.ones(N)
for b in range(INTRO_BARS, int(OUTRO_START)):
    for q in range(4):
        st = b * BAR + q * BEAT
        n = int(0.42 * SR)
        t = np.arange(n) / SR
        f = 48 + 110 * np.exp(-t * 28)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7.5)
        s += 0.25 * hp(rng.standard_normal(n), 3000) * np.exp(-t * 90)
        add(s, st, gain=0.55)
        i = int(st * SR)
        d = 1 - 0.55 * np.exp(-np.arange(int(0.35 * SR)) / SR * 9)
        duck[i:i + len(d)] = np.minimum(duck[i:i + len(d)], d[: max(0, N - i)])
L *= duck
R *= duck

# clap on 2 and 4
for b in range(INTRO_BARS + 1, int(OUTRO_START)):
    for q in (1, 3):
        n = int(0.3 * SR)
        t = np.arange(n) / SR
        s = hp(lp(rng.standard_normal(n), 6000), 900) * (np.exp(-t * 22) + 0.4 * np.exp(-((t - 0.012) ** 2) / 2e-5))
        add(s, b * BAR + q * BEAT, pan=0.05, gain=0.16)

# hats: closed 8ths, open on off-beats every other bar
for b in range(INTRO_BARS, int(OUTRO_START)):
    for e in range(8):
        n = int((0.18 if e % 2 else 0.05) * SR)
        t = np.arange(n) / SR
        s = lp(hp(rng.standard_normal(n), 7000), 11000) * np.exp(-t * (18 if e % 2 else 70))
        add(s, b * BAR + e * BEAT / 2, pan=0.35, gain=0.032 if e % 2 else 0.022)

# plucky arpeggio, 16ths, ping-pong pan
for b in range(INTRO_BARS + 2, n_bars):
    ch = CHORDS[b % len(CHORDS)]
    pattern = [ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[1] + 12, ch[0] + 24, ch[2] + 12, ch[1] + 12, ch[2] + 12]
    gain = 0.07 if b < OUTRO_START else 0.045
    for s16 in range(16):
        n = int(BEAT / 4 * SR * 1.6)
        m = pattern[s16 % 8]
        s = saw(note_hz(m), n, detune=0.002)
        cutoff = 2400 + 1600 * np.sin(2 * np.pi * (b * 16 + s16) / 64)
        s = lp(s, cutoff) * env_adsr(n, a=0.002, d=0.12, s=0.15, r=0.08)
        add(s, b * BAR + s16 * BEAT / 4, pan=-0.45 if s16 % 2 else 0.45, gain=gain)

# riser into the groove and a soft impact at outro
n = int(BAR * SR)
t = np.arange(n) / SR
riser = hp(rng.standard_normal(n), 400) * (t / t[-1]) ** 2
riser = lp(riser, 5000)
add(riser, 0.0, gain=0.10)
n = int(3.0 * SR)
t = np.arange(n) / SR
impact = np.sin(2 * np.pi * 40 * t) * np.exp(-t * 2.2) + 0.3 * lp(rng.standard_normal(n), 900) * np.exp(-t * 3)
add(impact, OUTRO_T, gain=0.35)

# ------------------------------------------------------------------ mix: reverb, master, fades
ir_n = int(2.2 * SR)
t = np.arange(ir_n) / SR
ir = rng.standard_normal(ir_n) * np.exp(-t * 2.6)
ir = lp(ir, 5200)
ir /= np.sqrt(np.sum(ir ** 2))
wetL = fftconvolve(L, ir)[:N]
wetR = fftconvolve(R, np.roll(ir, 37))[:N]
L = L + 0.22 * wetL
R = R + 0.22 * wetR
mix = np.stack([L, R], axis=1)
mix = hp(mix.T, 28).T
# fade in/out
fi, fo = int(0.4 * SR), int(3.2 * SR)
mix[:fi] *= np.linspace(0, 1, fi)[:, None]
mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
# soft clip + normalise to -1 dBFS
mix = np.tanh(mix * 1.4) / np.tanh(1.4)
mix *= 10 ** (-1 / 20) / np.max(np.abs(mix))
(ROOT / "build").mkdir(exist_ok=True)
wavfile.write(ROOT / "build" / "music.wav", SR, (mix * 32767).astype(np.int16))
print("wrote build/music.wav", DUR, "s")
