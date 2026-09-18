"""Ears — microphone capture + speech-to-text hook.

capture(seconds) -> np float32 mono at 16 kHz (via sounddevice; a
subprocess arecord fallback keeps this alive on minimal boxes).

transcribe(wav_path) -> text via faster-whisper if installed
(pip install faster-whisper; tiny/base models run fine on CPU, so this
works even without the GPU). Without it, returns None — the companion
then relies on non-speech audio features in the soup instead.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 16000


def capture(seconds=4.0, device=None):
    """Record `seconds` of mono audio; returns (np.float32 array, sr)."""
    try:
        import sounddevice as sd
        rec = sd.rec(int(seconds * SR), samplerate=SR, channels=1,
                     dtype="float32", device=device)
        sd.wait()
        return rec[:, 0].copy(), SR
    except Exception:
        pass
    # arecord fallback (ALSA almost always present on Linux)
    if shutil.which("arecord"):
        fd, path = tempfile.mkstemp(prefix="ms_cap_", suffix=".wav")
        subprocess.run(["arecord", "-q", "-d", str(int(seconds)), "-r",
                        str(SR), "-f", "FLOAT_LE", "-c", "1", path],
                       check=True, timeout=seconds + 10)
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        Path(path).unlink()
        return data, sr
    raise RuntimeError("no mic backend: install sounddevice or arecord")


class Ears:
    """Speech recognition with graceful absence."""

    def __init__(self, model_size="base"):
        self.model_size = model_size
        self._model = None
        self.available = self._try_load()

    def _try_load(self):
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="auto",
                                       compute_type="auto")
            return True
        except Exception:
            return False

    def transcribe_file(self, wav_path, language=None):
        if not self.available:
            return None
        segs, _ = self._model.transcribe(str(wav_path), language=language,
                                         vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip() or None

    def listen_text(self, seconds=4.0):
        """Capture + transcribe in one call. Returns text or None."""
        x, sr = capture(seconds)
        fd, path = tempfile.mkstemp(prefix="ms_ear_", suffix=".wav")
        import soundfile as sf
        sf.write(path, x, sr)
        try:
            return self.transcribe_file(path)
        finally:
            Path(path).unlink()
