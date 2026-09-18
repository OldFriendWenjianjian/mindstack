"""Generate synthesized guitar practice audio for testing.

Produces a plucked-string (Karplus-Strong) rendering of an exercise JSON,
plus a deliberately wrong variant to test the error-detection path.
Ground truth comes from the exercise file itself.
"""
import json
import sys
import numpy as np
import soundfile as sf

SR = 44100


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def pluck(freq, dur, sr=SR, decay=0.996, bright=0.5):
    """Karplus-Strong plucked string."""
    n = int(dur * sr)
    period = int(round(sr / freq))
    rng = np.random.default_rng(int(freq * 100) % (2**31))
    # brighter = more high harmonics in the excitation
    buf = rng.uniform(-1, 1, period)
    if bright < 0.9:
        buf = np.convolve(buf, [0.5 + bright / 2, 0.5 - bright / 2], "same")
    out = np.empty(n)
    idx = 0
    prev = 0.0
    for i in range(n):
        cur = buf[idx]
        out[i] = cur
        buf[idx] = decay * 0.5 * (cur + prev)
        prev = cur
        idx = (idx + 1) % period
    # gentle pick attack envelope
    a = int(0.004 * sr)
    out[:a] *= np.linspace(0.3, 1.0, a)
    out[-int(0.05 * sr):] *= np.linspace(1, 0, int(0.05 * sr))
    return out


def render(events, sr=SR, total=None):
    if total is None:
        total = max(e["t"] + e.get("dur", 0.5) for e in events) + 1.0
    audio = np.zeros(int(total * sr))
    for e in events:
        midis = [e["midi"]] if "midi" in e else CHORD_VOICINGS.get(e.get("chroma"), [])
        for j, m in enumerate(midis):
            f = midi_to_hz(m)
            sig = pluck(f, e.get("dur", 1.0), sr)
            s = int(e["t"] * sr)
            # chords slightly staggered like a strum
            sig *= 0.5 / max(1, len(midis)) ** 0.5
            end = min(s + len(sig), len(audio))
            audio[s:end] += sig[: end - s]
    peak = np.abs(audio).max()
    if peak > 0:
        audio *= 0.85 / peak
    return audio


# common open chord voicings (midi numbers, low string first)
CHORD_VOICINGS = {
    "Em": [40, 47, 52, 55, 59, 64],
    "C":  [48, 52, 55, 60, 64],
    "G":  [43, 47, 50, 55, 59, 67],
    "Am": [45, 52, 57, 60, 64],
    "D":  [50, 57, 62, 66],
}


def main():
    ex_path, out_prefix = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "good"
    ex = json.load(open(ex_path))
    events = json.loads(json.dumps(ex["events"]))  # deep copy

    if mode == "wrong":
        for e in events:
            if "midi" in e:
                if e["midi"] == 70:
                    e["midi"] = 69  # Bb -> A: classic miss
                elif e["midi"] == 67:
                    e["midi"] = 65  # G -> F: fingering slip
            else:
                e["chroma"] = {"Em": "C", "C": "Em", "G": "D"}.get(e["chroma"], e["chroma"])
    elif mode == "late":
        for e in events:
            e["t"] += 0.25  # dragging behind the beat

    audio = render(events)
    sf.write(f"{out_prefix}_{mode}.wav", audio, SR)
    print(f"wrote {out_prefix}_{mode}.wav ({len(audio)/SR:.1f}s)")


if __name__ == "__main__":
    main()
