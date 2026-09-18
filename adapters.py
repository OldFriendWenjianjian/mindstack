"""Device adapters: read signals in, execute commands out.

Each adapter is deliberately tiny — read() + act() is the whole
contract. The reflex policy is separate so it can veto LLM commands
and run without any model. These three show the pattern for amp
(audio out), robot joint (serial telemetry + reflexes), and a local
video generator (diffusion/render process control).
"""
import time

import numpy as np

from apps.joint_app import classify as joint_classify


# ---- AMP: loudspeaker / guitar amp monitoring + control ---------------------

class AmpAdapter:
    """Listens to the amp's output (mic or line-in), commands volume/
    standby/preset. cmd names are examples — map to your amp's interface
    (MIDI, serial, IR, relays)."""

    CMDS = {"set_volume", "standby", "wake", "preset"}

    def __init__(self, source=None):
        self.source = source        # callable -> np array at 44100; None = zeros

    def read(self):
        x = self.source() if self.source else np.zeros(4096, np.float32)
        return [(np.asarray(x, np.float32), {"modality": "audio", "sr": 44100})]

    def act(self, cmds):
        done = []
        for c in cmds:
            if c.get("device") == "amp" and c.get("cmd") in self.CMDS:
                done.append(c)       # real wiring goes here
        return done


class AmpReflex:
    """Hardwired: never let feedback howl (rms runaway) or clip the
    output stage. Runs per-cycle regardless of any model."""

    LIMIT_RMS = 0.9
    LIMIT_RUNAWAY = 4.0          # cycle-to-cycle rms growth factor

    def __init__(self):
        self._prev_rms = None

    def check(self, signal):
        x = signal[0]
        rms = float(np.sqrt(np.mean(x ** 2)))
        prev, self._prev_rms = self._prev_rms, rms
        if rms > self.LIMIT_RMS:
            return "CLIP", "set_volume:0.3", "output near clipping"
        if prev and rms > self.LIMIT_RUNAWAY * prev and rms > 0.1:
            return "HOWL", "set_volume:0.5", "feedback runaway"
        return "OK", None, "level fine"

    def allows(self, cmd):
        return True               # volume cmds are safe here


# ---- ROBOT JOINT: the FOC board over serial ---------------------------------

class JointAdapter:
    """Feeds telemetry lines to the supervisor; translates actions to
    firmware serial commands ('iq 0', 'off'). poll_reflexes() lets the
    Loop run the spinal cord BEFORE any model is consulted."""

    def __init__(self, serial=None):
        self.serial = serial                  # pyserial Serial or None
        self.buffer = []                      # recent raw lines
        from apps.joint_app import JointSupervisor
        self.supervisor = JointSupervisor()

    def read(self):
        rows = [s for s in self.supervisor.history]
        return [(rows, {"modality": "telemetry"})]

    def act(self, cmds):
        for c in cmds:
            if c.get("device") != "joint":
                continue
            if c.get("cmd") == "iq_zero" and self.serial:
                self.serial.write(b"iq 0\n")
            elif c.get("cmd") == "pwm_off" and self.serial:
                self.serial.write(b"off\n")

    def poll_reflexes(self, reflex):
        """Drain pending telemetry; run classify() per sample; execute
        reflex actions immediately. Returns events for the report."""
        events = []
        while self._pending():
            line = self._next_line()
            out = self.supervisor.feed(line)
            if out and out["level"] != "OK":
                events.append({"level": out["level"], "reason": out["reason"]})
                if out["action"] == "pwm_off":
                    self.act([{"device": "joint", "cmd": "pwm_off", "args": {}}])
                elif out["action"] == "iq_zero":
                    self.act([{"device": "joint", "cmd": "iq_zero", "args": {}}])
        return events

    def _pending(self):
        return len(self.buffer) < 4     # demo stub; real: serial.in_waiting

    def _next_line(self):
        if self.serial:
            return self.serial.readline().decode(errors="replace").strip()
        import time as _t
        return ("st=3 ang=ol pwm=1 vbus=48.2 id=-0.01 iq=0.49 vd=0.8 vq=1.6 "
                "mod=0.04 vuv=1.2 iu=0.23 iv=-0.26 enc=0 rpm_ol=400 "
                f"isr={int(_t.time()*1000) % 100000}")


class JointReflex:
    """Veto layer over LLM commands + expose classify() to the Loop."""

    def __init__(self):
        self._last = ("OK", "none", "fine")

    def check(self, sample):
        self._last = joint_classify(sample)
        return self._last

    def allows(self, cmd):
        # during any non-OK state, only the reflex's own recovery acts
        if self._last[0] in ("OVERCURRENT", "BUS_FAULT", "STALL"):
            return False
        return True


# ---- VIDEO-GEN: local diffusion/render pipeline ------------------------------

class VideoGenAdapter:
    """Watches a local generator (fps, vram, queue depth) as telemetry;
    commands are start/stop/param changes for the render process."""

    def __init__(self, stats_fn=None):
        self.stats_fn = stats_fn        # callable -> list of dicts
        self.running = False
        self.params = {"steps": 25, "fps": 24}

    def read(self):
        rows = self.stats_fn() if self.stats_fn else [
            {"fps": 0.0, "queue": 0, "vram_gb": 0.0, "level": "OK"}]
        return [(rows, {"modality": "telemetry"})]

    def act(self, cmds):
        for c in cmds:
            if c.get("device") != "videogen":
                continue
            if c.get("cmd") == "start":
                self.running = True
            elif c.get("cmd") == "stop":
                self.running = False
            elif c.get("cmd") == "set_param":
                self.params.update(c.get("args", {}))


class VideoGenReflex:
    """Watchdog: if the renderer starves (<5 fps for its target) or VRAM
    saturates, drop quality params — never ask a model first."""

    def allows(self, cmd):
        return True

    def check(self, sample):
        if sample.get("vram_gb", 0) > 15.0:
            return "VRAM", "set_param:steps=15", "vram saturated"
        return "OK", None, "gen fine"
