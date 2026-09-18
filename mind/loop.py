"""The loop — ONE general-purpose mind cycle for every device.

    signals ──> soup (GPU/numpy) ──> cloud GLM-5.3-Flash ──> feedback ──> actions
                                        │
                                reflex policy (pure function,
                                overrides everything, no model)

An "app" is now just: (adapter, reflex_policy, system_prompt). The loop
is shared. The LLM sees only the soup; its feedback returns as a
strict JSON action list that adapters execute.

LLM output contract (violations are dropped, never guessed):
    {"thought": "...", "say": "...",
     "actions": [{"device": "...", "cmd": "...", "args": {...}}]}
"""
import json
import time

from senses import soup as soup_mod
from mind.client import ask


class Loop:
    def __init__(self, adapter, reflex, prompt, name="loop", voice=False,
                 max_say_chars=220):
        self.adapter = adapter      # read() -> [(data, meta)]; act(cmds)
        self.reflex = reflex        # allows(cmd) veto + check(sample)
        self.prompt = prompt        # system prompt for the cloud model
        self.name = name
        self.voice = voice          # True = speak `say` aloud via TTS
        self.max_say_chars = max_say_chars
        self.last_feedback = None

    # ---- one full cycle ----
    def cycle(self, question=None, say=True):
        t0 = time.time()
        out = {"loop": self.name, "backend": soup_mod.backend()}

        # 1. reflexes first: device state is checked pre-anything
        reflex_events = self.adapter.poll_reflexes(self.reflex) \
            if hasattr(self.adapter, "poll_reflexes") else []
        out["reflex"] = reflex_events

        # 2. read signals -> soup (+optional vision clip)
        signals = self.adapter.read()
        soups, clip = [], None
        for sig in signals:
            s, c = soup_mod.extract(sig)
            soups.append(s)
            if c:
                clip = c
        out["soup"] = soups
        if not soups:
            out["error"] = "no signals"
            return out

        # 3. ask the cloud model: soup text always, frames when present
        q = question or "Assess the situation and give feedback."
        content = []
        if clip:
            for b64 in clip:
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text",
                        "text": f"{q}\n\nFeature soup:\n{soup_mod.io_json(soups)}"})
        result = ask([self.system_message(), {"role": "user", "content": content}],
                     max_tokens=900)
        out["where"] = result.get("where")
        out["latency_s"] = round(time.time() - t0, 1)

        # 4. parse feedback: strict contract, drop on violation
        feedback = parse_feedback(result.get("text"))
        out["thought"] = feedback.get("thought")
        out["say"] = feedback.get("say")
        out["error"] = result.get("error") or feedback.get("_parse_error")

        # 5. act: reflex veto first, then LLM commands
        cmds = feedback.get("actions", [])
        allowed = [c for c in cmds if self.reflex.allows(c)]
        out["actions"] = allowed
        out["actions_blocked"] = [c for c in cmds
                                  if c not in allowed]
        if allowed:
            self.adapter.act(allowed)

        # 6. speak: route `say` through the voice backend when enabled
        if self.voice and out.get("say"):
            try:
                from senses.voice import speak
                text = out["say"][:self.max_say_chars]
                b, wav = speak(text, play_audio=False)
                out["voice"] = {"backend": b, "wav": wav}
                if say:
                    print(f"[{self.name} speaks] {text}")
            except Exception as e:
                out["voice"] = {"error": str(e)}

        if say and out.get("say"):
            print(f"[{self.name}] {out['say']}")
        self.last_feedback = out
        return out

    def system_message(self):
        return {"role": "system", "content": self.prompt + ACTION_CONTRACT}


ACTION_CONTRACT = """

You must answer with ONLY a JSON object (no markdown fence), shaped:
{"thought": "one sentence of reasoning",
 "say": "one short sentence for the human, friendly",
 "actions": [{"device": "name", "cmd": "verb", "args": {}}]}
Empty actions list is normal. Never invent device names beyond what the
soup describes; when unsure, act on nothing and explain in `say`."""


def parse_feedback(text):
    """Strict JSON parse; returns {} with _parse_error on violation."""
    if not text:
        return {"_parse_error": "empty reply"}
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            try:
                obj = json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                return {"_parse_error": "no JSON object in reply"}
        else:
            return {"_parse_error": "no JSON object in reply"}
    if not isinstance(obj, dict):
        return {"_parse_error": "reply not an object"}
    obj.setdefault("actions", [])
    return obj
