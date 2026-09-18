"""Regression: DSP + coach on synthetic takes with known ground truth.

Each take is generated from the exercise file itself (make_test_audio.py),
so pass/fail thresholds here are absolute, not hand-tuned to one sample.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from dsp.analysis import analyze      # noqa: E402
from coach.grade import grade         # noqa: E402


def setup_module(module):
    for prefix, schema in [("riff", "riff_schema.json"), ("chords", "chords_schema.json")]:
        for mode in ["good", "wrong", "late"]:
            subprocess.run(
                [sys.executable, str(HERE / "make_test_audio.py"),
                 str(HERE / "exercises" / schema),
                 str(HERE / "audio_test" / prefix), mode],
                check=True, capture_output=True)


def _grade(prefix, schema, variant):
    ex = json.load(open(HERE / "exercises" / schema))
    audio, sr = sf.read(HERE / "audio_test" / f"{prefix}_{variant}.wav")
    return grade(ex, analyze(audio.astype(np.float32), sr))


def test_riff_good_is_clean():
    v = _grade("riff", "riff_schema.json", "good")
    assert v["score"] == 100, f"score {v['score']}%, faults: {[e for e in v['events'] if e['status'] != 'ok']}"


def test_riff_pitch_errors_detected_as_wrong_fret():
    v = _grade("riff", "riff_schema.json", "wrong")
    assert v["score"] < 60
    statuses = {e["status"] for e in v["events"]}
    assert "wrong_fret" in statuses
    hint = next(e["hint"] for e in v["events"] if e["status"] == "wrong_fret")
    assert "one fret too low" in hint


def test_riff_late_detected_as_tempo_drift():
    v = _grade("riff", "riff_schema.json", "late")
    assert v["score"] < 60
    assert "tempo_hint" in v
    assert "drag" in v["tempo_hint"]
    assert v["tempo_drift_ms"] > 150


def test_chords_good_is_clean():
    v = _grade("chords", "chords_schema.json", "good")
    assert v["score"] == 100, f"score {v['score']}%, faults: {[e for e in v['events'] if e['status'] != 'ok']}"


def test_chords_wrong_detected():
    v = _grade("chords", "chords_schema.json", "wrong")
    assert v["score"] < 60
    assert any(e["status"] in ("wrong", "muddy") for e in v["events"])


def test_analysis_timing_accuracy():
    """Detected note onsets within 50 ms of ground truth on the good take."""
    ex = json.load(open(HERE / "exercises" / "riff_schema.json"))
    audio, sr = sf.read(HERE / "audio_test" / "riff_good.wav")
    res = analyze(audio.astype(np.float32), sr)
    for ev in ex["events"]:
        near = [n for n in res["notes"] if abs(n["t0"] - ev["t"]) < 0.2]
        assert near, f"no detected note near {ev['t']}s"
        best = min(near, key=lambda n: abs(n["t0"] - ev["t"]))
        assert abs(best["t0"] - ev["t"]) < 0.05, f"onset off by {best['t0']-ev['t']:.3f}s"
        assert abs(best["midi"] - ev["midi"]) <= 0.5, "pitch off by a semitone"


def test_pitch_tracker_recovers_all_riff_notes():
    ex = json.load(open(HERE / "exercises" / "riff_schema.json"))
    audio, sr = sf.read(HERE / "audio_test" / "riff_good.wav")
    res = analyze(audio.astype(np.float32), sr)
    assert len(res["notes"]) == len(ex["events"]), \
        f"expected {len(ex['events'])} notes, got {len(res['notes'])}"
