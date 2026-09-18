"""Cat companion app — watch a kitten, log her activity, read her mood.

Reflex layer: motion detection + region tracking decide when something
interesting is happening (the colliculus). Only then are frames grabbed
and sent to the vision model with a feline-behavior prompt (the cortex).
Quiet time costs nothing; a 24/7 puppy-cam costs no tokens while the
kitten sleeps.

Live mode reads a camera (V4L2 device or HTTP MJPEG/RTSP via ffmpeg);
batch mode analyzes a video file after the fact.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from senses.vision import region_of_motion, frame_jpeg_b64  # noqa: E402
from mind import client                                     # noqa: E402

CAT_PROMPT = """You are a calm, observant cat-behavior analyst. These frames show \
a kitten over a short window (timestamps in order; the hottest motion region is \
noted). Judge: her activity level, apparent emotional state (playful / alert / \
relaxed / anxious / agitated...), what she's doing, and anything concerning \
(lethargy, distress, danger). If no cat is visible, say exactly that. \
Answer in <=50 words, one line."""

MOOD_SCALE = {
    "playful": 2, "relaxed": 1, "alert": 1, "curious": 2,
    "anxious": -1, "agitated": -2, "distressed": -3, "lethargic": -1,
}


def analyze_window(frames, ts, hot_region):
    """One vision-model call for a burst of frames."""
    ts_text = ", ".join(f"{t:.0f}s" for t in ts)
    region_text = (f"hottest motion near grid cell "
                   f"row {hot_region[0]}, col {hot_region[1]}"
                   if hot_region else "no strong motion")
    prompt = f"{CAT_PROMPT}\nFrame timestamps: {ts_text}. {region_text}."
    out = client.ask_vision(prompt, frames)
    return out.get("text"), out.get("error")


def watch_stream(source, fps=2.0, size=256, still_sec=20.0, burst=4,
                 quiet_prompt_every=0, log=print, one_pass=False):
    """Generator: yields (t, {event:'burst'|'still', analysis, motion}).

    Watches a live stream; accumulates a burst of `burst` frames whenever
    motion stays hot, and reports a stillness event after `still_sec`
    seconds of quiet (cats sleep 16h/day — stillness is information too,
    but it's cheap: no model call unless quiet_prompt_every is set).
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {source}")
    interval = 1.0 / fps
    prev = None
    t0 = time.time()
    motion_run = []          # (t, energy, region) of current hot run
    quiet_since = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if one_pass:
                    break
                time.sleep(0.2)
                continue
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                              (size, size))
            t = time.time() - t0
            energy = 0.0
            region = None
            if prev is not None:
                region = region_of_motion(gray, prev)
                energy = region[2] if region else 0.0
            prev = gray

            if energy > 1.5:                     # hot: something is moving
                motion_run.append((t, gray, region))
                quiet_since = None
                if len(motion_run) >= burst:
                    frames = [frame_jpeg_b64(g) for _, g, _ in motion_run]
                    ts = [tt for tt, _, _ in motion_run]
                    hot = motion_run[-1][2]
                    an, err = analyze_window(frames, ts, hot)
                    yield t, {"event": "burst", "analysis": an,
                              "error": err, "motion": energy}
                    motion_run = []
            else:                                # quiet
                if motion_run:
                    frames = [frame_jpeg_b64(g) for _, g, _ in motion_run]
                    ts = [tt for tt, _, _ in motion_run]
                    an, err = analyze_window(frames, ts, motion_run[-1][2])
                    yield t, {"event": "burst", "analysis": an,
                              "error": err, "motion": motion_run[-1][1]}
                    motion_run = []
                if quiet_since is None:
                    quiet_since = t
                elif t - quiet_since >= still_sec:
                    yield t, {"event": "still", "analysis": None,
                              "motion": energy}
                    quiet_since = t
            if one_pass and t > 0:
                break
    finally:
        cap.release()


def watch_file(video_path, log=print):
    """Batch mode: scan a clip, grab bursts around active moments, analyze."""
    from senses.vision import VideoScan
    import numpy as np
    import cv2

    scan = VideoScan(video_path, fps=4.0, size=192)
    if not len(scan.times):
        return {"error": "no frames"}
    # adaptive threshold: a small subject far from the camera moves few
    # pixels, so use relative activity, not an absolute energy bar
    e = scan.energy
    floor = 0.3
    hot_thr = max(floor, float(np.percentile(e, 60)))
    active = [i for i in range(len(e)) if e[i] > hot_thr and e[i] > floor]
    # cluster active samples into windows
    windows = []
    for i in active:
        t = float(scan.times[i])
        if windows and t - windows[-1][1] < 1.0:
            windows[-1][1] = t
        else:
            windows.append([t, t])
    log(f"{len(windows)} active window(s) in {float(scan.times[-1]):.0f}s clip")
    events = []
    for w0, w1 in windows:
        frames = []
        for frac in (0.25, 0.6, 0.9):
            t = w0 + (w1 - w0) * frac
            ok, frame = grab_frame(video_path, t)
            if ok:
                frames.append(frame_jpeg_b64(frame))
        if not frames:
            continue
        an, err = analyze_window(frames, [w0 + (w1 - w0) * f for f in (0.25, 0.6, 0.9)], None)
        events.append({"t0": round(w0, 1), "t1": round(w1, 1),
                       "analysis": an, "error": err})
        log(f"  [{w0:5.1f}-{w1:5.1f}s] {an}")
    return {"video": str(video_path), "events": events}


def grab_frame(video_path, t):
    import subprocess, io
    import numpy as np
    from PIL import Image
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}",
           "-i", str(video_path), "-frames:v", "1", "-f", "rawvideo",
           "-pix_fmt", "gray", "-s", "256x256", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or len(r.stdout) < 256 * 256:
        return False, None
    return True, np.frombuffer(r.stdout[:256 * 256], np.uint8).reshape(256, 256)


def main():
    ap = argparse.ArgumentParser(description="watch the kitten")
    ap.add_argument("source", help="video file, /dev/videoN, or rtsp/http URL")
    ap.add_argument("--live", action="store_true", help="treat source as live camera")
    ap.add_argument("--still-sec", type=float, default=20.0)
    args = ap.parse_args()
    if args.live:
        for t, ev in watch_stream(args.source, still_sec=args.still_sec):
            if ev["event"] == "burst":
                print(f"[{t:6.1f}s] motion -> {ev['analysis'] or ev.get('error')}")
            else:
                print(f"[{t:6.1f}s] still ...")
    else:
        result = watch_file(args.source)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
