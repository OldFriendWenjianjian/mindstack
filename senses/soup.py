"""Feature soup — the GPU sensory frontend.

ONE generic extractor for all signal types. Raw streams in (audio,
frames, telemetry, whatever the device adapter passes), a dict of
normalized feature rows out. This is the only place that touches the
GPU, and it is deliberately model-lite: statistics, spectra, embeddings
if a model is loaded. The cloud LLM receives the soup, not the signals.

Design:
  - extract(signal) -> {"modality": ..., "features": {...}, "clip": b64?}
  - everything float-rounded to 3 decimals: cheap tokens, stable JSON
  - GPU backend (torch) auto-used when available; numpy path identical

A "signal" is (data, meta): data = np array (audio 1-D, video T×H×W×C
or list of frames, telemetry = list of sample dicts).
"""
import base64
import io

import numpy as np

TORCH = None


def _torch():
    """Lazy torch import; returns module or None."""
    global TORCH
    if TORCH is None:
        try:
            import torch
            TORCH = torch
        except ImportError:
            TORCH = False
    return TORCH or None


def backend():
    t = _torch()
    if t and t.cuda.is_available():
        return f"cuda:{t.cuda.get_device_name(0)}"
    if t:
        return "cpu-torch"
    return "numpy"


# ---- per-modality feature banks --------------------------------------------

def audio_soup(x, sr):
    """Audio: rms, spectral centroid/rolloff/flux, zero-crossings, band
    energies, and a coarse log-mel-ish spectrogram summary (8 bands × 8
    time bins)."""
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    zc = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    centroid = float((freqs * spec).sum() / (spec.sum() + 1e-9))
    # rolloff85 (clamped: pure silence pushes the index past the end)
    cum = np.cumsum(spec) / (spec.sum() + 1e-9)
    idx = min(int(np.searchsorted(cum, 0.85)), len(freqs) - 1)
    rolloff = float(freqs[idx])
    # flux
    mag = np.abs(np.fft.rfft(np.reshape(x[: (n // 1024) * 1024], (-1, 1024))
                             * np.hanning(1024), axis=1))
    flux = float(np.mean(np.maximum(0, np.diff(mag, axis=0)).sum(axis=1))) if len(mag) > 1 else 0.0
    # 8 log-band energies
    edges = np.array([30, 80, 200, 500, 1200, 3000, 7500, 12000, 20000])
    bands = []
    for i in range(8):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        bands.append(round(float(np.sqrt(np.mean(spec[m]) + 1e-9)), 3))
    # spectrogram patch 8x8 (log energy over 8 log-bands x 8 time frames)
    hop = max(1, (n // 1024) // 8)
    frames = mag[::hop][:8]
    patch = 20 * np.log10(frames + 1e-6)
    patch = np.clip((patch - patch.min()) / (np.ptp(patch) + 1e-6), 0, 1)
    if patch.shape[0] < 8:  # pad short signals to a full 8x8 patch
        patch = np.pad(patch, ((0, 8 - patch.shape[0]), (0, 0)))
    # bin 513 rfft bins into 8 log-spaced bands -> patch is exactly 8x8
    f_bins = np.fft.rfftfreq(1024, 1 / sr)
    edges = np.geomspace(30, min(sr / 2, 20000), 9)
    binned = np.zeros((patch.shape[0], 8))
    for i in range(8):
        m = (f_bins >= edges[i]) & (f_bins < edges[i + 1])
        if m.any():
            binned[:, i] = patch[:, m].mean(axis=1)
    patch = np.clip((binned - binned.min()) / (np.ptp(binned) + 1e-6), 0, 1)
    return {
        "rms": round(rms, 4), "zerocross": round(zc, 3),
        "centroid_hz": round(centroid), "rolloff85_hz": round(rolloff),
        "flux": round(flux, 3), "band_rms": bands,
        "spec8x8": [[round(float(v), 2) for v in row] for row in patch],
    }


def video_soup(frames, want_clip=True, clip_frames=3):
    """Video: motion energy curve, cuts count, brightness/contrast trend,
    hottest region trajectory, and (optionally) a few JPEG keyframes for
    the cloud vision model."""
    grays = [f if f.ndim == 2 else f.mean(axis=-1) for f in frames]
    diffs = [0.0] + [float(np.abs(grays[i].astype(np.float32)
                                 - grays[i - 1].astype(np.float32)).mean())
                     for i in range(1, len(grays))]
    bright = [float(g.mean()) for g in grays]
    h, w = grays[0].shape
    hot = []
    for i in range(1, len(grays)):
        d = np.abs(grays[i].astype(np.float32) - grays[i - 1].astype(np.float32))
        r, c = 2, 2
        best = 0.0
        for rr in range(4):
            for cc in range(4):
                e = d[rr * h // 4:(rr + 1) * h // 4, cc * w // 4:(cc + 1) * w // 4].mean()
                if e > best:
                    best, r, c = e, rr, cc
        hot.append([r, c, round(float(best), 1)])
    soup = {
        "frames": len(grays),
        "motion": [round(v, 1) for v in diffs[::max(1, len(diffs) // 16)]],
        "motion_mean": round(float(np.mean(diffs)), 2),
        "motion_peak": round(float(np.max(diffs)), 2),
        "cuts_est": int(sum(1 for v in diffs if v > 25)),
        "brightness": [round(v) for v in bright[::max(1, len(bright) // 12)]],
        "hot_region_path": hot[::max(1, len(hot) // 10)],
    }
    clip = None
    if want_clip and len(grays):
        idxs = np.linspace(0, len(frames) - 1, min(clip_frames, len(frames))).astype(int)
        clip = [frame_jpeg(frames[i]) for i in idxs]
    return soup, clip


def telemetry_soup(samples):
    """Telemetry rows (dicts): per-key min/med/max + level counts.
    Works for FOC joints, thermostats, anything tabular."""
    if not samples:
        return {}
    keys = set()
    for s in samples:
        keys.update(k for k, v in s.items() if isinstance(v, (int, float)))
    out = {"n": len(samples), "stats": {}}
    for k in sorted(keys):
        vals = [float(s[k]) for s in samples if k in s]
        out["stats"][k] = {
            "min": round(min(vals), 3), "med": round(float(np.median(vals)), 3),
            "max": round(max(vals), 3)}
    levels = {}
    for s in samples:
        lv = s.get("level")
        if lv:
            levels[lv] = levels.get(lv, 0) + 1
    if levels:
        out["levels"] = levels
    return out


def frame_jpeg(frame, quality=68):
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.convert("L").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def extract(signal):
    """Route (data, meta) to the right bank. meta: {modality, sr?, fps?}."""
    data, meta = signal
    mod = meta.get("modality", "audio")
    if mod == "audio":
        return {"modality": mod, "features": audio_soup(data, meta.get("sr", 44100))}, None
    if mod == "video":
        soup, clip = video_soup(data, want_clip=meta.get("want_clip", True))
        return {"modality": mod, "features": soup}, clip
    if mod == "telemetry":
        return {"modality": mod, "features": telemetry_soup(data)}, None
    raise ValueError(f"unknown modality: {mod}")


def to_text(soup):
    """Compact JSON text of the soup — this is what the LLM sees."""
    return base64.b64encode(b"").decode()[:0] + io_json(soup)


def io_json(soup):
    import json
    return json.dumps(soup, ensure_ascii=False, separators=(",", ":"))
