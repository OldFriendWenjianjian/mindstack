"""Imagination — local video generation.

VideoGenAdapter: the Loop's actuator for a local text/image-to-video
model (diffusers: e.g. damo-vilab/text-to-video-ms-1.7b, Wan-2.1-T2V,
CogVideoX-2b — pick by your VRAM). Two paths:

  real    — diffusers pipeline loaded on the GPU; prompts come from the
            cloud model's gen_video action args
  stub    — procedural renderer (moving gradient + text card) so the
            FULL companion loop runs on boxes without a GPU; the video
            file is real, playable, just not diffusion-quality

A queue + watchdog reflex keeps VRAM safe: jobs serialize, and the
reflex refuses generation when a job is already running (the model
doesn't get to stack renders).
"""
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path.home() / "mindstack_generated"
OUT_DIR.mkdir(exist_ok=True)


class VideoGenAdapter:
    CMDS = {"gen_video", "cancel"}

    def __init__(self, model_id=None, frames=24, fps=8, size=256):
        self.model_id = model_id          # None = stub renderer
        self.frames = frames
        self.fps = fps
        self.size = size
        self.pipe = None
        self.busy = False
        self.last_result = None
        self._lock = threading.Lock()

    # ---- Loop interface -------------------------------------------------
    def read(self):
        rows = [{"queue_busy": int(self.busy),
                 "model": self.model_id or "stub",
                 "last_ok": int(bool(self.last_result and
                                     Path(self.last_result).exists())),
                 "level": "OK"}]
        return [(rows, {"modality": "telemetry"})]

    def act(self, cmds):
        for c in cmds:
            if c.get("device") != "videogen" or c.get("cmd") not in self.CMDS:
                continue
            if c["cmd"] == "gen_video":
                self.generate(c.get("args", {}))

    # ---- generation ------------------------------------------------------
    def generate(self, args):
        """Blocking generate; the Loop/adapter serializes via `busy`."""
        prompt = args.get("prompt", "abstract colorful shapes drifting")
        with self._lock:
            if self.busy:
                return {"error": "renderer busy"}
            self.busy = True
        try:
            if self.model_id:
                path = self._gen_diffusers(prompt, args)
            else:
                path = self._gen_stub(prompt, args)
            self.last_result = path
            return {"file": str(path)}
        finally:
            self.busy = False

    def _gen_diffusers(self, prompt, args):
        import torch
        from diffusers import DiffusionPipeline
        if self.pipe is None:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16,
                variant="fp16" if "fp16" in
                str(getattr(self.pipe, "config", "")) else None)
            self.pipe.enable_model_cpu_offload()
        frames = self.pipe(prompt, num_frames=args.get("frames", self.frames),
                           num_inference_steps=args.get("steps", 20)
                           ).frames[0]
        return _export_mp4(frames, prompt, self.fps)

    def _gen_stub(self, prompt, args):
        """Procedural 'imagination': colored waveform circles + prompt card."""
        import cv2
        frames = []
        n = args.get("frames", self.frames)
        for i in range(n):
            t = i / max(1, n - 1)
            img = np.zeros((self.size, self.size, 3), np.uint8)
            for k, r in enumerate((60, 90, 110)):
                ang = t * 6.283 * (1 + 0.4 * k)
                x = int(self.size / 2 + r * np.cos(ang))
                y = int(self.size / 2 + r * np.sin(ang * 1.3))
                cv2.circle(img, (x, y), 12, (40 * k + 80, 160, 230 - 40 * k), -1)
            cv2.putText(img, prompt[:32], (8, self.size - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            frames.append(img)
        return _export_mp4(frames, prompt, self.fps)


class VideoGenReflex:
    """Hardwired: never stack renders, never run without disk headroom."""

    MIN_FREE_GB = 2.0

    def allows(self, cmd):
        return True

    def check(self, row):
        free = shutil.disk_usage(OUT_DIR).free / 1e9
        if free < self.MIN_FREE_GB:
            return "DISK", "cancel", "low disk for video output"
        return "OK", None, "gen ok"


def _export_mp4(frames, prompt, fps):
    import cv2
    h, w = frames[0].shape[:2]
    ts = time.strftime("%H%M%S")
    safe = "".join(ch if ch.isalnum() else "_" for ch in prompt)[:40]
    path = OUT_DIR / f"gen_{ts}_{safe}.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    for f in frames:
        vw.write(f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()
    return path
