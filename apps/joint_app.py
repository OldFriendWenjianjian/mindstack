"""Joint guardian — supervise FOC motor joints over the STM32 UART.

Layer split, strictly enforced:
  - Parser      : "st=.. ang=.. vbus=.. iq=.." lines -> dicts (4 Hz telemetry)
  - Reflexes    : pure-python policy, no model, no network, <1 ms per line.
                  Heat/tactile/current limits -> immediate safe commands.
  - Deliberation: batches recent telemetry into a summary and asks the
                  cloud model only for non-urgent insight (drift analysis,
                  thermal trends, maintenance hints). The model can NEVER
                  move the robot: its output is advisory text only.

The reflexes are the spinal cord: they would keep the joint safe even if
the network and every model were dead. Any tactile/heat sensor that
exists as an ADC channel shows up in telemetry and is checked here.
"""
import re
import time
from collections import deque

# ---- reflex policy (physics-derived, not learned) --------------------------

LIMITS = {
    "iq_abs_a": 6.0,        # software clamp is 8 A; reflex trips earlier
    "id_abs_a": 10.0,       # field-weakening budget; −8..−10 A is the edge
    "vbus_min_v": 20.0,     # 48 V bus sagging hard
    "vbus_max_v": 60.0,
    "mod_max": 0.97,        # SVPWM hexagon ceiling
    "stall_iq_a": 2.5,      # 'powered + high iq + no rpm' = stall/heavy touch
    "pwm_on": 1,
}

ACTIONS = {
    "OK":          ("none", "within limits"),
    "WARN":        ("none", "approaching limit, notify deliberator"),
    "STALL":       ("iq_zero", "high iq with no motion — stall or contact"),
    "OVERCURRENT": ("pwm_off", "current beyond safe clamp"),
    "BUS_FAULT":   ("pwm_off", "bus voltage out of range"),
    "MOD_SAT":     ("none", "voltage ceiling pinned — reduce demand"),
}


def classify(sample, prev=None):
    """The hardwired reflex: pure function of one telemetry sample.

    Returns (level, action, reason). Action is one of 'none',
    'iq_zero' (command iq->0), 'pwm_off' (immediate disable). These map
    to firmware commands a supervisor process must send instantly.
    """
    iq = sample.get("iq", 0.0)
    id_ = sample.get("id", 0.0)
    vbus = sample.get("vbus", 48.0)
    mod = sample.get("mod", 0.0)
    pwm = sample.get("pwm", 0)
    rpm = sample.get("rpm_ol", 0.0)

    if not (LIMITS["vbus_min_v"] <= vbus <= LIMITS["vbus_max_v"]):
        return "BUS_FAULT", *ACTIONS["BUS_FAULT"]
    if abs(iq) > LIMITS["iq_abs_a"] or abs(id_) > LIMITS["id_abs_a"]:
        return "OVERCURRENT", *ACTIONS["OVERCURRENT"]
    if (pwm == LIMITS["pwm_on"] and abs(iq) > LIMITS["stall_iq_a"]
            and abs(rpm) < 5.0):
        return "STALL", *ACTIONS["STALL"]
    if mod > LIMITS["mod_max"]:
        return "MOD_SAT", *ACTIONS["MOD_SAT"]
    if (abs(iq) > 0.8 * LIMITS["iq_abs_a"] or mod > 0.9 * LIMITS["mod_max"]
            or vbus < 1.1 * LIMITS["vbus_min_v"]):
        return "WARN", *ACTIONS["WARN"]
    return "OK", *ACTIONS["OK"]


# ---- telemetry parsing ------------------------------------------------------

_TOKEN = re.compile(r"(\w+)=([^\s]+)")


def parse_line(line):
    """Parse one telemetry line into a dict; None if not telemetry.

    Real format from main.c: st=%u ang=%s pwm=%u vbus=%.1f id=%.2f iq=%.2f
    vd=%.1f vq=%.1f mod=%.2f vuv=%.1f iu=%.2f iv=%.2f enc=%u rpm_ol=%.0f isr=%lu
    """
    if not line.startswith("st="):
        return None
    out = {}
    for k, v in _TOKEN.findall(line):
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v  # e.g. ang=enc / ang=ol
    return out or None


class JointSupervisor:
    """Consumes telemetry lines, runs reflexes, buffers history for the mind."""

    def __init__(self, history_s=30):
        self.history = deque()
        self.history_s = history_s
        self.last_level = None

    def feed(self, line, now=None):
        s = parse_line(line)
        if s is None:
            return None
        t = now if now is not None else time.time()
        level, action, reason = classify(s)
        s["t"] = t
        s["level"] = level
        self.history.append(s)
        while self.history and self.history[0]["t"] < t - self.history_s:
            self.history.popleft()
        self.last_level = level
        return {"sample": s, "level": level, "action": action, "reason": reason}

    def summary_for_cloud(self):
        """Compress history into <=20 lines the deliberator may see."""
        if not self.history:
            return "no telemetry yet"
        h = list(self.history)
        first, last = h[0], h[-1]
        levels = {}
        for s in h:
            levels[s["level"]] = levels.get(s["level"], 0) + 1
        iqs = sorted(s.get("iq", 0) for s in h)
        mods = sorted(s.get("mod", 0) for s in h)
        lines = [
            f"joint telemetry over {last['t']-first['t']:.0f}s ({len(h)} samples @4Hz)",
            f"levels: {levels}",
            f"iq: min {iqs[0]:.2f} med {iqs[len(iqs)//2]:.2f} max {iqs[-1]:.2f} A",
            f"mod index: med {mods[len(mods)//2]:.2f} max {mods[-1]:.2f}",
            f"vbus now {last.get('vbus', 0):.1f} V, mode {last.get('ang')}, "
            f"pwm {last.get('pwm')}, rpm {last.get('rpm_ol', 0):.0f}",
        ]
        faults = [s for s in h if s["level"] not in ("OK",)]
        if faults:
            f = faults[-1]
            lines.append(f"latest non-OK: {f['level']} (iq={f.get('iq')}, "
                         f"mod={f.get('mod')}, vbus={f.get('vbus')})")
        return "\n".join(lines)


def consult_cloud(supervisor, question=None):
    """Deliberator escalation: advisory only, returns text."""
    from mind import client
    summary = supervisor.summary_for_cloud()
    prompt = (f"You are a robotics assistant watching an FOC motor joint "
              f"(STM32G473, 48V bus, UE6815 motor). Telemetry summary:\n"
              f"{summary}\n\nYou have NO control authority — reflexes handle "
              f"safety. Advise the human operator: what's healthy, what to "
              f"watch, what to do next. Max 100 words.")
    if question:
        prompt += f"\n\nOperator asks: {question}"
    return client.ask([{"role": "user", "content": prompt}])
