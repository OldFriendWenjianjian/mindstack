#!/usr/bin/env python3
"""guitar-teacher CLI.

  pass <exercise.json> <take.wav>       analyze one recorded pass
  live <exercise.json>                  coach a live loop (Ctrl-C to stop)
  ask "<question>"                      ask the cloud teacher (with history)
  demo                                  run all synthetic test takes

Layers:
  DSP (reflex)   -> notes/onsets/chords at audio rate, no LLM
  Coach (rules)  -> instant verdict + hints, no LLM
  Local LLM      -> optional spoken-tone phrasing, if llama-server/Ollama up
  Cloud GLM      -> the deliberative teacher, gets coach summaries only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from dsp.analysis import analyze            # noqa: E402
from coach.grade import grade, summarize_for_cloud  # noqa: E402
from coach import local_phrase              # noqa: E402
from cloud.teacher import ask_cloud, SessionLog  # noqa: E402


def do_pass(ex_path, wav_path, log, speak=False):
    ex = json.load(open(ex_path))
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    res = analyze(audio.astype(np.float32), sr)
    verdict = grade(ex, res)

    print(f"\n=== {verdict['exercise']} — score {verdict['score']}% ===")
    for e in verdict["events"]:
        mark = {"ok": " ✓", "missed": " ✗", "timing": " ⏱", "pitch": " ♪",
                "wrong_fret": " ✗", "wrong_note": " ✗", "muddy": " ~",
                "wrong": " ✗"}.get(e["status"], " ?")
        line = f"{mark} {e.get('label', '?')}"
        if e["status"] == "ok":
            print(line)
        else:
            print(f"{line}  [{e['status']}] {e.get('hint', '')}")
    for x in verdict.get("extra_notes", []):
        print(f"  + extra {x['name']} @ {x['t']}s — {x['hint']}")
    if "tempo_hint" in verdict:
        print(f"  TEMPO: {verdict['tempo_hint']}")

    # optional local phrasing of the top faults (never blocks anything)
    if speak and local_phrase.available():
        hints = [e["hint"] for e in verdict["events"] if e.get("hint")]
        spoken = local_phrase.phrase(hints)
        if spoken:
            print(f"\n[local coach] {spoken}")

    log.append({"kind": "pass", "verdict": verdict, "wav": str(wav_path)})
    return verdict


def do_live(ex_path, log):
    """Loop: record chunks from the default mic, coach each pass.
    Requires sounddevice + PortAudio; falls back with instructions."""
    try:
        import sounddevice as sd
    except ImportError:
        print("live mode needs: pip install soundfilter sounddevice  "
              "(and PortAudio); for now use: pass <exercise> <take.wav>")
        return
    ex = json.load(open(ex_path))
    dur = max(e["t"] + e.get("dur", 1.0) for e in ex["events"]) + 1.0
    sr = 22050
    print(f"live: play '{ex['name']}' ({dur:.0f}s), Ctrl-C to stop")
    n = 0
    try:
        while True:
            n += 1
            print(f"\n--- pass {n}: recording {dur:.0f}s ...")
            buf = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            res = analyze(buf[:, 0], sr)
            verdict = grade(ex, res)
            print(f"score {verdict['score']}%")
            for e in verdict["events"]:
                if e["status"] != "ok":
                    print(f"  {e.get('label','?')}: [{e['status']}] {e.get('hint','')}")
            log.append({"kind": "pass", "verdict": verdict, "pass_n": n})
    except KeyboardInterrupt:
        print("\nstopped")


def do_ask(question, log):
    history = log.history_for_cloud()
    out = ask_cloud("general chat with your guitar teacher", question, history)
    if "error" in out:
        print(f"cloud error: {out['error']}")
        return
    print(f"\n[cloud teacher, {out['latency_s']}s]\n{out['reply']}")
    log.append({"kind": "ask", "question": question, "reply": out["reply"]})


def do_demo(log):
    here = Path(__file__).parent
    cases = [("exercises/riff_schema.json", "audio_test/riff_good.wav"),
             ("exercises/riff_schema.json", "audio_test/riff_wrong.wav"),
             ("exercises/riff_schema.json", "audio_test/riff_late.wav"),
             ("exercises/chords_schema.json", "audio_test/chords_good.wav"),
             ("exercises/chords_schema.json", "audio_test/chords_wrong.wav")]
    for ex_rel, wav_rel in cases:
        do_pass(here / ex_rel, here / wav_rel, log)


def main():
    ap = argparse.ArgumentParser(description="local+cloud guitar teacher")
    ap.add_argument("cmd", choices=["pass", "live", "ask", "demo"])
    ap.add_argument("a", nargs="?", help="exercise.json / question")
    ap.add_argument("b", nargs="?", help="take.wav")
    ap.add_argument("--speak", action="store_true",
                    help="route fault lines through the local LLM if present")
    ap.add_argument("--log", default=str(Path(__file__).parent / "session.jsonl"))
    args = ap.parse_args()

    log = SessionLog(args.log)
    if args.cmd == "pass":
        if not (args.a and args.b):
            sys.exit("usage: pass <exercise.json> <take.wav>")
        do_pass(args.a, args.b, log, speak=args.speak)
    elif args.cmd == "live":
        do_live(args.a, log)
    elif args.cmd == "ask":
        if not args.a:
            sys.exit('usage: ask "why do my chords buzz?"')
        do_ask(args.a, log)
    elif args.cmd == "demo":
        do_demo(log)


if __name__ == "__main__":
    main()
