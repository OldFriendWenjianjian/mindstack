#!/usr/bin/env python3
"""tiny_demo.py — the smallest complete MindStack application.

The whole flow in ~70 lines, one cycle, every stage printed:

    signal -> soup -> cloud GLM -> JSON feedback -> reflex veto -> action

This demo uses a synthetic 1-second "microphone" signal so it runs on
any machine with no hardware. To use your real microphone:
    pip install sounddevice
    python3 tiny_demo.py --mic

Only requirement: a cloud key (export GLM_API_KEY=...).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mind.loop import Loop                                    # noqa: E402
from senses import soup as soup_mod                           # noqa: E402

SR = 22050


# ---------- 1. the device: a mic that is either real or synthetic ------------

def make_signal(use_mic):
    if use_mic:
        import sounddevice as sd
        x = sd.rec(SR, samplerate=SR, channels=1, dtype="float32")[:, 0]
        import time; time.sleep(SR / SR + 0.1)
        print("[device] recorded 1 s from your microphone")
        return x
    t = np.arange(SR) / SR
    x = (0.4 * np.sin(2 * np.pi * 220 * t)          # a 220 Hz tone...
         + 0.1 * np.random.randn(SR)).astype(np.float32)  # ...with noise
    print("[device] synthetic mic: 220 Hz tone + noise (no hardware needed)")
    return x


# ---------- 2. adapter: read() gives signals, act() executes commands --------

class TinyAdapter:
    def __init__(self, use_mic):
        self.x = make_signal(use_mic)
        self.volume = 0.5

    def read(self):
        return [(self.x, {"modality": "audio", "sr": SR})]

    def act(self, cmds):
        for c in cmds:
            if c["cmd"] == "set_volume":
                self.volume = float(c["args"].get("v", self.volume))
            print(f"[actuator] {c['device']}.{c['cmd']}({c['args']}) "
                  f"-> volume now {self.volume}")


# ---------- 3. reflex: hardwired, no model, vetoes unsafe commands -----------

class LoudnessReflex:
    def __init__(self, adapter):
        self.adapter = adapter

    def rms(self):
        return float(np.sqrt(np.mean(self.adapter.x ** 2)))

    def allows(self, cmd):
        loud = self.rms() > 0.7
        if loud and cmd["cmd"] == "set_volume" and float(cmd["args"].get("v", 0)) > 0.5:
            print(f"[reflex] VETO {cmd} — signal too loud ({self.rms():.2f})")
            return False
        return True


# ---------- 4. assemble the loop and run ONE cycle ----------------------------

PROMPT = ("You control exactly one device: a speaker named 'speaker'. "
          "You receive a feature soup of 1 second of room audio from it. "
          "Keep volume comfortable: send {'device':'speaker','cmd':'set_volume',"
          "'args':{'v':0.0-1.0}} whenever it should change; do nothing if "
          "the level is fine. Answer only via the JSON contract.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true", help="use real microphone")
    args = ap.parse_args()

    adapter = TinyAdapter(args.mic)
    loop = Loop(adapter, LoudnessReflex(adapter), PROMPT, name="tiny")

    print(f"\n[stage 0] soup backend: {soup_mod.backend()}")
    print(f"[stage 0] signal rms: {loop.reflex.rms():.3f}\n")

    out = loop.cycle(question="What do you hear? Should the volume change?")

    print(f"\n[stage 1] soup sent to cloud:\n  {soup_mod.io_json(out['soup'])[:180]} ...")
    print(f"\n[stage 2] cloud ({out['where']}, {out['latency_s']}s) thought: {out['thought']}")
    print(f"[stage 2] says: {out['say']}")
    print(f"[stage 3] executed: {out['actions']} | vetoed: {out['actions_blocked']}")
