"""Vision sense — the retina for cameras and movies.

Cheap, model-free, real-time primitives:
- downsampled frame stream
- motion energy per frame (mean abs diff of grayscale)
- scene-cut detection (histogram distance spike)
- activity segments (contiguous above/below-threshold spans)

Everything downstream (movie describer, cat watcher) consumes the
segment list, not raw frames — tokens stay bounded regardless of
video length.
"""
import subprocess
from pathlib import Path

import numpy as np


def probe_duration(video_path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def sample_frames(video_path, fps=2.0, size=256, start=0.0, duration=None):
    """Yield (t_sec, gray_frame) at a fixed low sample rate."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(video_path), "-vf",
            f"fps={fps},scale={size}:{size},format=gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = size * size
    idx = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        yield start + idx / fps, np.frombuffer(buf, np.uint8).reshape(size, size)
        idx += 1
    proc.wait()


class VideoScan:
    """Motion-energy timeline + scene cuts for one video."""

    def __init__(self, video_path, fps=2.0, size=256):
        self.path = Path(video_path)
        self.fps = fps
        times, energies, cuts = [], [], []
        prev = None
        prev_hist = None
        for t, frame in sample_frames(self.path, fps, size):
            gray = frame.astype(np.float32)
            if prev is not None:
                energies.append(float(np.abs(gray - prev).mean()))
            else:
                energies.append(0.0)
            prev = gray
            # scene cut: coarse histogram L1 distance spike
            hist = np.histogram(frame, bins=16, range=(0, 255))[0].astype(np.float32)
            hist /= hist.sum() + 1e-9
            if prev_hist is not None:
                d = float(np.abs(hist - prev_hist).sum())
                cuts.append((t, d))
            prev_hist = hist
            times.append(t)
        self.times = np.array(times)
        self.energy = np.array(energies)
        self.cuts = self._pick_cuts(cuts)

    def _pick_cuts(self, pairs):
        """Cut = histogram jump that's both absolute and relative.

        The distance distribution is near-zero-variance for static-ish
        content, so a pure sigma rule collapses; require an absolute
        floor plus 8x the running typical distance.
        """
        if not pairs:
            return []
        ds = np.array([d for _, d in pairs])
        typical = float(np.median(ds)) + 1e-6
        cut_ts = []
        for (t, d) in pairs:
            if d > 0.25 and d > 8 * typical:
                if not cut_ts or t - cut_ts[-1] > 1.0:
                    cut_ts.append(t)
        return cut_ts

    def activity_segments(self, min_gap=1.5, min_len=0.0):
        """Split the timeline into segments by scene cuts + activity gaps.

        Returns [{t0, t1, mean_motion, peak_motion}]. These are the units
        the describer sends to the model — 'shots', roughly.
        """
        bounds = [0.0]
        for c in self.cuts:
            bounds.append(c)
        end = float(self.times[-1]) if len(self.times) else 0.0
        segs = []
        # also split where motion stays near zero for > min_gap
        quiet = self.energy < max(0.5, np.percentile(self.energy, 10))
        run_start = None
        for i, q in enumerate(quiet):
            if q and run_start is None:
                run_start = self.times[i]
            elif not q and run_start is not None:
                if self.times[i] - run_start >= min_gap:
                    bounds.append(float(run_start))
                run_start = None
        bounds = sorted(set(bounds))
        for i, b0 in enumerate(bounds):
            b1 = bounds[i + 1] if i + 1 < len(bounds) else end + 1.0 / self.fps
            mask = (self.times >= b0) & (self.times < b1)
            if not mask.any():
                continue
            e = self.energy[mask]
            if (b1 - b0) < min_len:
                continue
            segs.append({
                "t0": round(float(b0), 2), "t1": round(float(b1), 2),
                "mean_motion": round(float(e.mean()), 2),
                "peak_motion": round(float(e.max()), 2),
            })
        return segs


def region_of_motion(frame_gray, prev_gray, grid=4):
    """Which grid cell holds the most motion — coarse attention/gaze.

    Returns (row, col, energy) of the hottest cell, or None.
    This is the 'superior colliculus': it decides where to look.
    """
    if prev_gray is None or frame_gray.shape != prev_gray.shape:
        return None
    h, w = frame_gray.shape
    gh, gw = h // grid, w // grid
    diff = np.abs(frame_gray.astype(np.float32) - prev_gray.astype(np.float32))
    best, best_e = None, 0.0
    for r in range(grid):
        for c in range(grid):
            e = float(diff[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw].mean())
            if e > best_e:
                best, best_e = (r, c), e
    if best is None or best_e < 1.0:
        return None
    return best[0], best[1], round(best_e, 2)


def frame_jpeg_b64(frame_gray, quality=70):
    """Encode a gray frame as base64 JPEG for a vision model."""
    import base64
    import io

    from PIL import Image
    img = Image.fromarray(frame_gray)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
