#!/usr/bin/env python3
"""companion_demo.py — the full companion in one file.

    camera + microphone  ->  soup  ->  cloud GLM  ->  voice + actions
                                              |->  local video generation

Hardware-optional: with no camera/mic it uses synthetic senses (the
cloud still sees real soups); with no GPU the video generator renders
procedural mp4s instead of diffusion. Voice uses flite/espeak by
default — install piper for a natural voice.

  python3 companion_demo.py                 # fully synthetic, safe anywhere
  python3 companion_demo.py --cam 0 --mic   # real senses (your box)
  python3 companion_demo.py --gen-model damo-vilab/text-to-video-ms-1.7b
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mind.loop import Loop                                        # noqa: E402
from senses import soup as soup_mod                               # noqa: E402

SR = 16000


# ---------- senses ------------------------------------------------------------

def get_signals(args):
    signals = []
    if args.cam is not None:
        from senses.eyes import CameraAdapter
        cam = CameraAdapter(args.cam, window_s=2.0)
        signals += cam.read()
        print(f"[eyes] camera {args.cam}: {signals[-1][0].__len__()} frames")
    else:
        frames = [np.full((64, 64, 3), int(60 + 40 * np.sin(i / 3)), np.uint8)
                  for i in range(8)]
        signals.append((frames, {"modality": "video", "want_clip": True}))
        print("[eyes] synthetic frames (use --cam 0 for real camera)")

    heard = None
    if args.mic:
        from senses.ears import capture
        x, sr = capture(3.0)
        heard = f"{len(x)/sr:.1f}s audio"
        print(f"[ears] microphone: {heard}")
    else:
        t = np.arange(SR * 2) / SR
        x = (0.3 * np.sin(2 * np.pi * 220 * t)
             + 0.05 * np.random.randn(SR * 2)).astype(np.float32)
        print("[ears] synthetic tone (use --mic for real microphone)")
    signals.append((x, {"modality": "audio", "sr": SR}))

    # optional: speech-to-text joins the question if a whisper model exists
    if args.mic:
        try:
            from senses.ears import Ears
            text = Ears("tiny").listen_text(3.0)
            if text:
                print(f"[ears] you said: {text!r}")
                signals[-1][1]["transcript"] = text
        except Exception as e:
            print(f"[ears] ASR unavailable ({type(e).__name__}) — continuing")
    return signals


# ---------- adapter: everything the mind may act on ---------------------------

class CompanionAdapter:
    def __init__(self, args):
        self.args = args
        self.gen = None
        if args.gen:                       # video generation enabled
            from senses.imagination import VideoGenAdapter
            self.gen = VideoGenAdapter(model_id=args.gen_model or None)
        self.artifacts = []

    def read(self):
        return get_signals(self.args)

    def act(self, cmds):
        for c in cmds:
            dev, cmd = c.get("device"), c.get("cmd")
            if self.gen and dev == "videogen" and cmd == "gen_video":
                prompt = c.get("args", {}).get("prompt", "abstract dream")
                print(f"[imagination] generating: {prompt!r}")
                r = self.gen.generate({"prompt": prompt})
                if "file" in r:
                    self.artifacts.append(r["file"])
                    print(f"[imagination] -> {r['file']}")
                else:
                    print(f"[imagination] failed: {r}")


# ---------- reflexes -----------------------------------------------------------

class CompanionReflex:
    def allows(self, cmd):
        # only devices we wired may be commanded, whatever the model says
        ok = cmd.get("device") in ("videogen",)
        if not ok:
            print(f"[reflex] VETO unknown device: {cmd}")
        return ok


PROMPT = """You are a friendly desktop companion with senses: you receive a \
feature soup of what a camera and microphone just observed (and optionally a \
transcript of speech). React like a warm, curious being: one or two sentences \
in `say` (this will be spoken aloud). If the moment would make a nice short \
animation, add action {"device":"videogen","cmd":"gen_video","args":{"prompt":\
"<english visual description, 5-12 words>"}} — at most one per reply."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=None, help="camera index (e.g. 0)")
    ap.add_argument("--mic", action="store_true", help="use real microphone")
    ap.add_argument("--gen", action="store_true", help="enable video generation")
    ap.add_argument("--gen-model", default=None,
                    help="diffusers model id (default: procedural stub)")
    ap.add_argument("--no-voice", action="store_true")
    args = ap.parse_args()

    adapter = CompanionAdapter(args)
    loop = Loop(adapter, CompanionReflex(), PROMPT, name="companion",
                voice=not args.no_voice)

    out = loop.cycle(question="Look and listen. What's happening? React, and "
                              "animate the mood if you like.")
    print(f"\n[soup] {soup_mod.io_json(out['soup'])[:160]} ...")
    print(f"[cloud] {out['where']} in {out['latency_s']}s")
    print(f"[thought] {out['thought']}")
    print(f"[say] {out['say']}")
    print(f"[actions] executed={out['actions']} blocked={out['actions_blocked']}")
    if out.get("voice"):
        v = out["voice"]
        print(f"[voice] {v.get('backend')}: {v.get('wav', v.get('error'))}")
    for a in adapter.artifacts:
        print(f"[video] {a}")


if __name__ == "__main__":
    main()
