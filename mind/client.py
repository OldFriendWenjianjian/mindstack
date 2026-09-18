"""Mind client — one interface to every LLM layer.

`ask()` routes to the local model (if LOCAL_LLM_URL is set and alive)
or straight to the cloud. `ask_cloud()` always goes to the cloud GLM.
Apps never talk HTTP themselves.
"""
import base64
import json
import os
from pathlib import Path

import requests

ZCODE_CFG = Path("/root/.zcode/cli/config.json")
CLOUD_BASE = "https://api.niumacode.cc/v1"
CLOUD_MODEL = "glm-5.3-flash"
LOCAL_BASE = os.environ.get("LOCAL_LLM_URL", "")
LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen3:14b")


def _api_key():
    key = os.environ.get("GLM_API_KEY")
    if key:
        return key
    try:
        return json.load(open(ZCODE_CFG))["provider"]["niumacode"]["options"]["apiKey"]
    except Exception:
        return None


def local_available(timeout=0.8):
    if not LOCAL_BASE:
        return False
    try:
        requests.get(f"{LOCAL_BASE}/models", timeout=timeout)
        return True
    except Exception:
        return False


def _post(base, model, messages, max_tokens, temperature, timeout):
    """Chat completion with reasoning-budget handling; returns text or None."""
    key = _api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = requests.post(
        f"{base}/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    choice = r.json()["choices"][0]
    content = choice.get("message", {}).get("content")
    if not content and choice.get("finish_reason") == "length" and max_tokens < 12000:
        # reasoning consumed the budget; retry with headroom
        return _post(base, model, messages, max_tokens * 3, temperature, timeout)
    return content


def ask(messages, max_tokens=2000, temperature=0.6, prefer_local=False, timeout=120):
    """Route to local model when available+preferred, else cloud."""
    if prefer_local and LOCAL_BASE and local_available():
        try:
            text = _post(LOCAL_BASE, LOCAL_MODEL, messages, max_tokens, temperature, timeout)
            if text:
                return {"text": text, "where": "local", "model": LOCAL_MODEL}
        except Exception:
            pass  # fall through to cloud
    key = _api_key()
    if not key:
        return {"text": None, "where": "none",
                "error": "no API key: set GLM_API_KEY or ZCode config"}
    try:
        text = _post(CLOUD_BASE, CLOUD_MODEL, messages, max_tokens, temperature, timeout)
        return {"text": text, "where": "cloud", "model": CLOUD_MODEL}
    except Exception as e:
        return {"text": None, "where": "none", "error": str(e)}


def ask_vision(prompt, jpeg_b64_frames, max_tokens=1500, temperature=0.4, timeout=180):
    """Send up to a few JPEG frames (base64) + prompt to the cloud vision model."""
    content = []
    for b64 in jpeg_b64_frames:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    return ask([{"role": "user", "content": content}],
               max_tokens=max_tokens, temperature=temperature, timeout=timeout)
