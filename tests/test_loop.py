"""Generic loop tests: soup extraction, feedback contract, reflex veto.

Cloud is stubbed — these verify the architecture's mechanics, not GLM.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from senses import soup as soup_mod                      # noqa: E402
from mind.loop import Loop, parse_feedback               # noqa: E402


# ---- soup extraction --------------------------------------------------------

def test_audio_soup_shape():
    sr = 22050
    t = np.arange(sr) / sr
    x = (100 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    s, clip = soup_mod.extract((x, {"modality": "audio", "sr": sr}))
    f = s["features"]
    assert clip is None
    assert 0 < f["centroid_hz"] < 2000
    assert len(f["band_rms"]) == 8
    assert len(f["spec8x8"]) == 8 and len(f["spec8x8"][0]) == 8


def test_video_soup_and_clip():
    frames = [np.full((64, 64), v, np.uint8) for v in range(0, 200, 20)]
    s, clip = soup_mod.extract((frames, {"modality": "video"}))
    f = s["features"]
    assert f["frames"] == 10
    assert len(clip) == 3                 # keyframes for cloud vision
    assert "motion_mean" in f


def test_telemetry_soup_stats():
    rows = [{"iq": 0.4, "vbus": 48.0, "level": "OK"},
            {"iq": 5.0, "vbus": 47.9, "level": "WARN"}]
    s, _ = soup_mod.extract((rows, {"modality": "telemetry"}))
    st = s["features"]["stats"]
    assert st["iq"]["min"] == 0.4 and st["iq"]["max"] == 5.0
    assert s["features"]["levels"] == {"OK": 1, "WARN": 1}


def test_backend_reports():
    assert isinstance(soup_mod.backend(), str)


# ---- feedback contract ------------------------------------------------------

def test_parse_clean_json():
    fb = parse_feedback('{"thought":"t","say":"s","actions":[{"device":"amp","cmd":"set","args":{"v":3}}]}')
    assert fb["actions"][0]["device"] == "amp"
    assert fb["_parse_error" in fb] if "_parse_error" in fb else True


def test_parse_fenced_and_embedded_json():
    fenced = '```json\n{"thought":"t","say":"s","actions":[]}\n```'
    assert parse_feedback(fenced)["actions"] == []
    embedded = 'Sure! {"thought":"t","say":"s","actions":[]} hope that helps'
    assert parse_feedback(embedded)["actions"] == []


def test_parse_garbage_is_flagged_not_crashed():
    fb = parse_feedback("I am sorry, I cannot")
    assert fb.get("_parse_error")
    fb = parse_feedback(None)
    assert fb.get("_parse_error")


# ---- the loop with a stubbed cloud ------------------------------------------

class StubAdapter:
    """One adapter = read signals + execute commands. That's all."""

    def __init__(self):
        self.executed = []

    def read(self):
        x = np.zeros(4096, np.float32)
        return [(x, {"modality": "audio", "sr": 22050})]

    def act(self, cmds):
        self.executed.extend(cmds)


class AllowAllReflex:
    def allows(self, cmd):
        return True

    def check(self, sample):
        return "OK", "none", "fine"


def test_loop_full_cycle_with_stub(monkeypatch):
    adapter = StubAdapter()
    loop = Loop(adapter, AllowAllReflex(), "You are a bench assistant.", "stub")
    reply = json.dumps({
        "thought": "signal is silence",
        "say": "All quiet.",
        "actions": [{"device": "amp", "cmd": "standby", "args": {}}]})
    monkeypatch.setattr("mind.loop.ask", lambda *a, **k: {"text": reply, "where": "stub"})
    out = loop.cycle()
    assert out["thought"] == "signal is silence"
    assert adapter.executed[0]["cmd"] == "standby"
    assert out["error"] is None or out["error"] == ... or out["error"] is None


def test_loop_garbage_reply_acts_on_nothing(monkeypatch):
    adapter = StubAdapter()
    loop = Loop(adapter, AllowAllReflex(), "p", "stub2")
    monkeypatch.setattr("mind.loop.ask",
                        lambda *a, **k: {"text": None, "where": "stub",
                                         "error": "cloud down"})
    out = loop.cycle()
    assert adapter.executed == []          # failed mind -> no action
    assert out["error"]


def test_loop_drops_commands_reflex_vetoes(monkeypatch):
    class VetoReflex(AllowAllReflex):
        def allows(self, cmd):
            return cmd.get("cmd") != "danger"

    adapter = StubAdapter()
    loop = Loop(adapter, VetoReflex(), "p", "stub3")
    reply = json.dumps({"thought": "t", "say": "s", "actions": [
        {"device": "amp", "cmd": "ok_thing", "args": {}},
        {"device": "amp", "cmd": "danger", "args": {}}]})
    monkeypatch.setattr("mind.loop.ask", lambda *a, **k: {"text": reply, "where": "stub"})
    out = loop.cycle()
    assert [c["cmd"] for c in adapter.executed] == ["ok_thing"]
    assert len(out["actions_blocked"]) == 1
