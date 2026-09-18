"""DSP layer — the "retina" of the guitar teacher.

Pure-numpy/scipy, real-time capable primitives:
- YIN pitch tracker (f0 per frame)
- spectral-flux onset detector
- chroma (12-dim pitch-class energy) for chord identification
- note tracking (onset + pitch -> note events)

No model weights, no GPU: this runs in the reflex layer at audio rate.
"""
import numpy as np
from scipy import signal

SR = 22050
FRAME = 1024          # ~46 ms
HOP = 256             # ~11.6 ms — pitch update rate ~86 Hz
FMIN = 70.0           # low E is 82.4 Hz
FMAX = 700.0          # 22nd fret high E ~ 660 Hz+ headroom
NOTE_MIN, NOTE_MAX = 39, 76   # E2..E5

# chroma filterbank: 2 harmonics per pitch class, tight bands
CHROMA_EDGES = []
for pc in range(12):
    edges = []
    for h in (1, 2):
        f_lo = 55.0 * 2 ** (pc / 12) * h * 2 ** (-1 / 48)
        f_hi = 55.0 * 2 ** (pc / 12) * h * 2 ** (1 / 48)
        edges.append((f_lo, f_hi))
    CHROMA_EDGES.append(edges)


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(f / 440.0)


def yin_pitch(frame, sr=SR, fmin=FMIN, fmax=FMAX, thresh=0.15):
    """YIN fundamental frequency estimate for one frame. Returns Hz or 0."""
    w = len(frame)
    half = w // 2
    x = frame - frame.mean()
    # difference function
    d = np.zeros(half)
    for lag in range(1, half):
        v = x[: w - lag] - x[lag:]
        d[lag] = np.dot(v, v)
    if d[1:].sum() == 0:
        return 0.0
    # cumulative mean normalized difference
    cmnd = np.ones(half)
    run = 0.0
    for lag in range(1, half):
        run += d[lag]
        cmnd[lag] = d[lag] * lag / run if run > 0 else 1.0
    # absolute threshold: first dip below thresh
    tau = 0
    for lag in range(2, half - 1):
        if cmnd[lag] < thresh:
            while lag + 1 < half - 1 and cmnd[lag + 1] < cmnd[lag]:
                lag += 1
            tau = lag
            break
    if tau == 0:
        # fall back to global minimum if it's decent
        lag = int(np.argmin(cmnd[2:]) ) + 2
        if cmnd[lag] < 0.35:
            tau = lag
        else:
            return 0.0
    # parabolic interpolation
    if 1 <= tau < half - 1:
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = a + c - 2 * b
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        tau = tau + shift
    f = sr / tau
    return f if fmin <= f <= fmax else 0.0


def spectral_flux(stream, sr=SR, n_fft=1024):
    """Positive spectral flux envelope + its adaptive threshold."""
    f, t, Z = signal.stft(stream, sr, window="hann", nperseg=n_fft,
                          noverlap=n_fft - 256, padded=False, boundary=None)
    mag = np.abs(Z)
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)
    # insert leading zero to align with t
    flux = np.concatenate([[0.0], flux])
    # adaptive threshold: median + delta over 0.4 s window
    k = max(3, int(0.4 / (n_fft - 256) * sr) | 1)
    pad = np.pad(flux, k // 2, mode="edge")
    med = np.array([np.median(pad[i:i + k]) for i in range(len(flux))])
    thr = med + 1.5 * flux.std() + 1e-9
    peaks = flux > thr
    onsets = []
    i = 0
    tt = t[: len(flux)]
    while i < len(flux):
        if peaks[i]:
            j = i
            while j + 1 < len(flux) and peaks[j + 1]:
                j += 1
            onsets.append(tt[i + np.argmax(flux[i:j + 1])])
            i = j + 1
        else:
            i += 1
    # enforce 60 ms minimum inter-onset
    out = []
    for o in onsets:
        if not out or o - out[-1] > 0.06:
            out.append(o)
    return np.array(out), flux, tt[: len(flux)]


def chroma_frame(frame, sr=SR):
    """12-dim chroma energy for one frame (A=0 ... G#=11 by pc index)."""
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    freqs = np.fft.rfftfreq(len(frame), 1 / sr)
    chroma = np.zeros(12)
    for pc, edges in enumerate(CHROMA_EDGES):
        e = 0.0
        for f_lo, f_hi in edges:
            band = (freqs >= f_lo) & (freqs < f_hi)
            e += spec[band].sum()
        chroma[pc] = e
    tot = chroma.sum()
    return chroma / tot if tot > 0 else chroma


PC_NAMES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def chroma_to_names(chroma):
    """Map pc-index chroma (A=0) to name-keyed dict."""
    return {PC_NAMES[i]: float(chroma[i]) for i in range(12)}


def track_notes(stream, sr=SR, min_dur=0.06):
    """YIN pitch over frames -> discrete note segments (midi, t0, t1)."""
    frames = []
    for i in range(0, len(stream) - FRAME, HOP):
        f = yin_pitch(stream[i:i + FRAME], sr)
        frames.append(f)
    # convert to midi, median filter to kill octave jumps
    midi = np.array([hz_to_midi(f) if f > 0 else np.nan for f in frames])
    notes = []
    cur = None
    for i, m in enumerate(midi):
        if np.isnan(m):
            if cur and (i * HOP / sr - cur[1]) >= min_dur:
                notes.append(cur)
            cur = None
            continue
        if cur and abs(m - cur[0]) < 0.7:
            # update running mean pitch
            cur[3] += [m]
            cur[0] = float(np.median(cur[3]))
            cur[2] = i * HOP / sr + FRAME / sr
        else:
            if cur and (i * HOP / sr - cur[1]) >= min_dur:
                notes.append(cur)
            cur = [float(m), i * HOP / sr, i * HOP / sr + FRAME / sr, [m]]
    if cur and (len(stream) / sr - cur[1]) >= min_dur:
        notes.append(cur)
    return [{"midi": int(round(n[0])), "t0": n[1], "t1": n[2], "mid": np.mean(n[3])}
            for n in notes if NOTE_MIN - 1 <= n[0] <= NOTE_MAX + 1]


def analyze(stream, sr=SR):
    """Full analysis: notes + onsets + mean chroma around each onset window."""
    notes = track_notes(stream, sr)
    onsets, _, _ = spectral_flux(stream, sr)
    # chroma over 0.35 s window after each onset
    chords = []
    w = int(0.35 * sr)
    for o in onsets:
        s = int(o * sr)
        seg = stream[s:s + w]
        if len(seg) < w // 2:
            continue
        chords.append({"t": float(o), "chroma": chroma_to_names(chroma_frame(seg, sr))})
    return {"notes": notes, "onsets": [float(o) for o in onsets], "chords": chords}
