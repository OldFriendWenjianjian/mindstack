"""Voice output — the mouth. speak(text) -> audio on the speaker.

Backends, best first:
  piper   — neural TTS, natural voice, needs `pip install piper-tts`
            plus a voice model (~60 MB); recommended on the GPU box
  flite   — small classic TTS, apt install flite, no model download
  espeak-ng — last-resort robotic voice, almost always installed

All backends render to a WAV file, then play via ffplay/ffmpeg (present
everywhere). If aplay exists it is preferred (lower latency, no ffprobe).
"""
import os
import shutil
import subprocess
import tempfile
import time


def backend():
    if shutil.which("piper") or _piper_py():
        return "piper"
    if shutil.which("flite"):
        return "flite"
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return "espeak-ng"
    return None


def _piper_py():
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


def render_wav(text, out_path, voice=None):
    """Synthesize text to out_path with the best available backend."""
    b = backend()
    if b == "piper":
        _render_piper(text, out_path, voice)
    elif b == "flite":
        cmd = ["flite", "-t", text, "-o", out_path]
        if voice:
            cmd[1:1] = ["-voice", voice]
        subprocess.run(cmd, check=True, timeout=60)
    elif b == "espeak-ng":
        subprocess.run(["espeak-ng", "-w", out_path, text], check=True, timeout=60)
    else:
        raise RuntimeError("no TTS backend found (flite / espeak-ng / piper)")
    return b


def _render_piper(text, out_path, voice):
    exe = shutil.which("piper")
    if exe:
        cmd = ["piper", "--output_file", out_path]
        if voice:
            cmd += ["--model", voice]
        subprocess.run(cmd, input=text.encode(), check=True, timeout=120)
    else:
        from piper import PiperVoice
        pv = PiperVoice.load(voice) if voice else PiperVoice.load()
        with open(out_path, "wb") as f:
            pv.synthesize(text, f)


def play(path):
    """Play a wav via aplay if present, else ffplay (quiet)."""
    if shutil.which("aplay"):
        subprocess.run(["aplay", "-q", path], check=False, timeout=300)
    elif shutil.which("ffplay"):
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel",
                        "quiet", path], check=False, timeout=300)
    else:
        raise RuntimeError("no audio player (aplay / ffplay)")


def speak(text, voice=None, play_audio=True):
    """One-shot: text in, voice out. Returns (backend, wav_path)."""
    fd, path = tempfile.mkstemp(prefix="mindstack_tts_", suffix=".wav")
    os.close(fd)
    b = render_wav(text, path, voice)
    if play_audio:
        play(path)
    return b, path


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "MindStack voice output is online."
    t0 = time.time()
    b, p = speak(text)
    print(f"[voice] backend={b} {time.time()-t0:.1f}s -> {p}")
