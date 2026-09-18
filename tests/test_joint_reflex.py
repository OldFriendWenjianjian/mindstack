"""Reflex + parser tests for joint_app against the REAL firmware format."""
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from apps.joint_app import JointSupervisor, classify, parse_line  # noqa: E402

# exact line shape from firmware/src/main.c telemetry()
REAL_LINE = ("st=3 ang=ol pwm=1 vbus=48.2 id=-0.01 iq=0.49 vd=0.8 vq=1.6 "
             "mod=0.04 vuv=1.2 iu=0.23 iv=-0.26 enc=0 rpm_ol=400 isr=48123")


def test_parse_real_firmware_line():
    s = parse_line(REAL_LINE)
    assert s is not None
    assert s["st"] == 3
    assert s["ang"] == "ol"          # string token, not numeric
    assert s["pwm"] == 1
    assert abs(s["vbus"] - 48.2) < 0.01
    assert abs(s["iq"] - 0.49) < 0.01
    assert s["isr"] == 48123
    assert isinstance(s["st"], int)


def test_parse_ignores_non_telemetry():
    assert parse_line("boot") is None
    assert parse_line("help:") is None
    assert parse_line("") is None


def test_healthy_sample_is_ok():
    s = parse_line(REAL_LINE)
    level, action, reason = classify(s)
    assert level == "OK"
    assert action == "none"


def test_overcurrent_trips_pwm_off():
    s = parse_line(REAL_LINE)
    s["iq"] = 7.0                     # above 6 A reflex limit, below 8 A clamp
    level, action, _ = classify(s)
    assert level == "OVERCURRENT"
    assert action == "pwm_off"


def test_stall_detection():
    s = parse_line(REAL_LINE)
    s["iq"] = 3.0                     # > 2.5 A
    s["rpm_ol"] = 0                   # not turning
    s["pwm"] = 1
    level, action, _ = classify(s)
    assert level == "STALL"
    assert action == "iq_zero"


def test_bus_fault_low_and_high():
    s = parse_line(REAL_LINE)
    s["vbus"] = 12.0
    assert classify(s)[0] == "BUS_FAULT"
    s["vbus"] = 65.0
    assert classify(s)[0] == "BUS_FAULT"


def test_mod_saturation_warns():
    s = parse_line(REAL_LINE)
    s["mod"] = 0.99
    assert classify(s)[0] == "MOD_SAT"


def test_warn_band():
    s = parse_line(REAL_LINE)
    s["iq"] = 5.0                     # 80-100% of limit, not above
    assert classify(s)[0] == "WARN"


def test_supervisor_history_and_summary():
    sup = JointSupervisor()
    t0 = 1000.0
    for i in range(40):               # 10 s of 4 Hz telemetry
        out = sup.feed(REAL_LINE, now=t0 + i * 0.25)
        assert out["level"] in ("OK", "WARN")
    assert len(sup.history) == 40
    summary = sup.summary_for_cloud()
    assert "40 samples" in summary
    assert "iq:" in summary
    # a fault appears in the summary
    bad = REAL_LINE.replace("iq=0.49", "iq=7.5")
    sup.feed(bad, now=t0 + 11)
    assert "OVERCURRENT" in sup.summary_for_cloud()


def test_history_window_prunes():
    sup = JointSupervisor(history_s=5)
    t0 = 2000.0
    for i in range(60):               # 15 s at 4 Hz -> only ~5 s retained
        sup.feed(REAL_LINE, now=t0 + i * 0.25)
    assert len(sup.history) <= 21
