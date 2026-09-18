"""Movie companion app — watch a video, understand story, emotion, content.

Pipeline: vision scan -> shot segments -> keyframes per segment ->
one vision-model call per segment (bounded tokens) -> storyline merge
by the text model. Long videos stay cheap: you pay per shot, not per
frame, and the merge sees shot descriptions, not pixels.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from senses.vision import VideoScan, sample_frames, frame_jpeg_b64  # noqa: E402
from mind import client                                            # noqa: E402

SHOT_PROMPT = """You are watching one shot (segment) of a video. The frames are \
samples, in order. Describe: what happens, the setting, any people/animals and \
their apparent emotions, the mood/atmosphere, and one line on how it might fit \
a larger story. Max 60 words."""

STORY_PROMPT = """These are chronological shot descriptions of one video, with \
timestamps. Write: (1) a 3-5 sentence storyline of the whole video, (2) the \
overall emotional arc (start -> end), (3) the 2 most notable moments with \
timestamps. Be concrete; don't invent details the shots don't support."""


def grab_frames(video_path, t0, t1, n=3, size=320):
    """n JPEG frames spread across [t0, t1] via per-frame ffmpeg seek."""
    frames = []
    for i in range(n):
        t = t0 + (t1 - t0) * (i + 0.5) / n
        out = subprocess_run_frame(video_path, t, size)
        if out is not None:
            frames.append(out)
    return frames


def subprocess_run_frame(video_path, t, size):
    import subprocess
    import io
    from PIL import Image
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", f"{t:.3f}", "-i", str(video_path), "-frames:v", "1",
           "-vf", f"scale={size}:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    img = Image.open(io.BytesIO(r.stdout))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def describe_segment(video_path, seg, max_frames=3):
    frames = grab_frames(video_path, seg["t0"], seg["t1"], n=max_frames)
    if not frames:
        return {"segment": seg, "description": "(unreadable segment)"}
    ts = ", ".join(f"{seg['t0'] + (seg['t1']-seg['t0'])*(i+0.5)/max_frames:.1f}s"
                   for i in range(len(frames)))
    prompt = SHOT_PROMPT + f"\nFrame timestamps: {ts}. Segment motion: mean {seg['mean_motion']}, peak {seg['peak_motion']}."
    out = client.ask_vision(prompt, frames)
    desc = out.get("text") or f"(model unavailable: {out.get('error')})"
    return {"segment": seg, "description": desc}


def understand(video_path, max_segments=12, log=print):
    scan = VideoScan(video_path, fps=2.0)
    segs = scan.activity_segments()
    log(f"video: {video_path}")
    log(f"scene cuts at {[c for c in scan.cuts]}s, {len(segs)} segments")
    if len(segs) > max_segments:
        # keep the most active segments, chronologically ordered
        segs = sorted(segs, key=lambda s: -s["mean_motion"])[:max_segments]
        segs = sorted(segs, key=lambda s: s["t0"])

    shots = []
    for i, seg in enumerate(segs):
        log(f"  shot {i+1}/{len(segs)} [{seg['t0']}-{seg['t1']}s] ...")
        shots.append(describe_segment(video_path, seg))

    timeline = "\n".join(
        f"[{s['segment']['t0']}-{s['segment']['t1']}s] {s['description']}"
        for s in shots)
    story = client.ask([{"role": "user", "content": f"{STORY_PROMPT}\n\n{timeline}"}],
                       max_tokens=1200)
    return {
        "video": str(video_path),
        "cuts": scan.cuts,
        "shots": shots,
        "storyline": story.get("text"),
        "storyline_error": story.get("error"),
    }


def main():
    ap = argparse.ArgumentParser(description="watch a movie with the mind stack")
    ap.add_argument("video")
    ap.add_argument("--max-shots", type=int, default=12)
    args = ap.parse_args()
    result = understand(args.video, args.max_shots)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
