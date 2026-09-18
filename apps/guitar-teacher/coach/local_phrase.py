"""Optional local phrasing layer (the 'subcortical' LLM on your own box).

If a local OpenAI-compatible server (llama.cpp llama-server / Ollama) is
running, instant feedback lines get rephrased by the local model in a
friendly voice. If it's absent, the coach's rule-based hints are already
good enough and everything still works. Cloud is never blocked by this.
"""
import os

import requests

LOCAL_BASE = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1")
LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen3:14b")

SYSTEM = (
    "You rephrase a guitar coach's technical notes into ONE short, friendly "
    "spoken line (max 25 words). No lists, no markdown, talk like a teacher "
    "sitting next to the student."
)


def available(timeout=0.8):
    try:
        requests.get(f"{LOCAL_BASE}/models", timeout=timeout)
        return True
    except Exception:
        return False


def phrase(hints, timeout=8):
    """Rephrase a list of rule-based hints; returns text or None on failure."""
    if not hints:
        return None
    try:
        r = requests.post(
            f"{LOCAL_BASE}/chat/completions",
            json={"model": LOCAL_MODEL,
                  "messages": [
                      {"role": "system", "content": SYSTEM},
                      {"role": "user", "content": "; ".join(hints[:3])},
                  ],
                  "max_tokens": 60, "temperature": 0.7},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None
