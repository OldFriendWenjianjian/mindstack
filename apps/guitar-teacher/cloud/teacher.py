"""Cloud teacher — the slow deliberative layer (GLM via OpenAI-compatible API).

Never sees audio: receives the coach's structured verdict summary plus
practice history, and returns teacher-level guidance. API key is read
from the ZCode config (same endpoint ZCode itself uses) or GLM_API_KEY.
"""
import base64
import json
import os
import time
from pathlib import Path

import requests

ZCODE_CFG = Path("/root/.zcode/cli/config.json")
DEFAULT_BASE = "https://api.niumacode.cc/v1"
DEFAULT_MODEL = "glm-5.3-flash"

SYSTEM_PROMPT = """You are a patient, encouraging guitar teacher. You receive a student's \
practice-pass report produced by an automatic listener (pitch/timing/chord analysis). \
The student is a beginner working through riffs and open chords.

Reply with:
1. One-sentence verdict of the pass.
2. The single most important thing to fix next (be concrete: string, fret, hand).
3. A short drill for it (30-60 seconds long).
Keep it under 120 words, warm but honest. If the pass was clean, say so and \
raise the challenge slightly (tempo, dynamics, or a new exercise)."""


def _api_key():
    key = os.environ.get("GLM_API_KEY")
    if key:
        return key
    try:
        return json.load(open(ZCODE_CFG))["provider"]["niumacode"]["options"]["apiKey"]
    except Exception:
        return None


def ask_cloud(summary, question=None, history=None, model=DEFAULT_MODEL, timeout=120):
    """Send a coach summary (and optional student question) to the cloud model.

    The endpoint puts visible text in `content` and thinking in
    `reasoning_content`; at max reasoning effort the thinking can consume
    the whole max_tokens budget, returning content=None with
    finish_reason=length. We budget generously and retry once bigger.
    """
    key = _api_key()
    if not key:
        return {"error": "no API key: set GLM_API_KEY or install ZCode config"}

    content = f"Practice pass report:\n{summary}"
    if question:
        content += f"\n\nStudent asks: {question}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": content})

    t0 = time.time()
    reply = None
    max_tokens = 4000
    for attempt in range(2):
        r = requests.post(
            f"{DEFAULT_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens,
                  "temperature": 0.6},
            timeout=timeout,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        choice = r.json()["choices"][0]
        msg = choice.get("message", {})
        reply = msg.get("content")
        if reply:
            break
        if choice.get("finish_reason") == "length":
            max_tokens *= 3   # reasoning ate the budget; retry with more room
            continue
        break
    dt = time.time() - t0
    if not reply:
        return {"error": "model produced no visible answer (reasoning exhausted "
                         "or empty response); try again or reduce reasoning effort"}
    return {"reply": reply, "latency_s": round(dt, 1), "model": model}


class SessionLog:
    """Append-only JSONL history of passes and cloud replies."""

    def __init__(self, path="session.jsonl"):
        self.path = Path(path)

    def append(self, record):
        record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def history_for_cloud(self, last=6):
        """Recent pass scores + key faults, as chat messages for context."""
        out = []
        if not self.path.exists():
            return out
        for line in open(self.path).read().splitlines()[-last:]:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "verdict" in r:
                faults = [e for e in r["verdict"].get("events", []) if e["status"] != "ok"]
                msg = (f"Earlier pass of '{r['verdict']['exercise']}': "
                       f"score {r['verdict'].get('score')}%, "
                       f"{len(faults)} faults")
                if faults:
                    msg += f", worst: {faults[0].get('hint', faults[0]['status'])}"
                out.append({"role": "user", "content": msg})
            elif "reply" in r:
                out.append({"role": "assistant", "content": r["reply"][:300]})
        return out
