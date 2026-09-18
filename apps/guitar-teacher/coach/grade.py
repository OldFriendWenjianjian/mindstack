"""Coach layer — the "cerebellum": deterministic, instant, rule-based.

Diffs detected notes/chords against the exercise and emits structured
feedback events. No LLM involved; this must feel immediate. The output
dict is also what gets summarized for the cloud teacher, so the LLM
never sees raw audio — only the coach's distilled verdicts.
"""
import json
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def load_exercise(path):
    return json.load(open(path))


def _match_notes(truth, detected, tol_ms, tol_cents):
    """Two-stage matching.

    Strict pass: pitch AND time inside tolerance -> "ok" candidate.
    Lenient pass: leftovers matched within 350 ms regardless of pitch,
    so errors surface as wrong-note / wrong-timing instead of
    "missed + extra" noise. Remaining truth = truly missed.
    """
    LENT_MS = 350
    pairs = []
    used = set()

    def nearest(t, max_dt, need_pitch):
        best, best_cost = None, None
        for i, d in enumerate(detected):
            if i in used:
                continue
            dt = (d["t0"] - t["t"]) * 1000
            if abs(dt) > max_dt:
                continue
            cents = abs(d["mid"] - t["midi"]) * 100
            if need_pitch and cents > tol_cents:
                continue
            cost = abs(dt) + cents
            if best_cost is None or cost < best_cost:
                best, best_cost = i, cost
        return best

    for t in truth:
        i = nearest(t, tol_ms, True)
        if i is not None:
            used.add(i)
            d = detected[i]
            pairs.append({"event": t, "match": "ok",
                          "dt_ms": (d["t0"] - t["t"]) * 1000,
                          "cents": (d["mid"] - t["midi"]) * 100})
        else:
            pairs.append({"event": t, "match": "none"})

    for p in pairs:
        if p["match"] == "ok":
            continue
        i = nearest(p["event"], LENT_MS, False)
        if i is not None:
            used.add(i)
            d = detected[i]
            p["match"] = "lenient"
            p["dt_ms"] = (d["t0"] - p["event"]["t"]) * 1000
            p["cents"] = (d["mid"] - p["event"]["midi"]) * 100
            p["played_midi"] = int(round(d["midi"]))

    extras = [d for i, d in enumerate(detected) if i not in used]
    return pairs, extras


def _match_chords(truth, detected, tol_ms):
    """Coverage/leakage matching: fraction of chroma energy on chord tones.

    Real chroma never matches voicing-count ratios (string loudness varies),
    so we score like a teacher would: are the right notes present and the
    wrong ones absent, not exact energy ratios.
    """
    pairs = []
    used = set()
    for t in truth:
        target = t["chroma_target"]
        tones = {k for k, w in target.items() if w > 0.2}
        best, best_cov = None, None
        for i, c in enumerate(detected):
            if i in used or abs(c["t"] - t["t"]) > tol_ms / 1000 + 0.5:
                continue
            tot = sum(c["chroma"].values())
            if tot <= 0:
                continue
            cov = sum(c["chroma"].get(k, 0) for k in tones) / tot
            if best_cov is None or cov > best_cov:
                best, best_cov = i, cov
        if best is not None:
            used.add(best)
            pairs.append({"event": t, "coverage": best_cov,
                          "chroma": detected[best]["chroma"], "tones": tones})
        else:
            pairs.append({"event": t, "missed": True})
    return pairs


def _chord_hint(tones, chroma):
    """Name the most audible offender / absentee for a chord problem."""
    tot = sum(chroma.values())
    if tot <= 0:
        return None
    inside = sum(chroma.get(k, 0) for k in tones) / tot
    if inside < 0.65:
        # loudest off-chord pitch class is the story
        off = {k: v for k, v in chroma.items() if k not in tones}
        bad = max(off, key=off.get)
        if off[bad] / tot > 0.15:
            return f"'{bad}' is loud but doesn't belong in this chord — wrong shape or a ringing open string"
        return "chord shape unrecognizable — re-fret slowly and strum one string at a time"
    # enough coverage but some chord tone is missing
    weak = min(tones, key=lambda k: chroma.get(k, 0) / tot)
    if chroma.get(weak, 0) / tot < 0.10:
        return f"'{weak}' is barely sounding — that string is muted or not fretted"
    return "a bit muddy — let the old chord go before grabbing the next"


def grade(exercise, analysis):
    """Produce the structured session verdict."""
    tol_ms = exercise.get("tolerance_ms", 120)
    tol_cents = exercise.get("tolerance_cents", 60)
    verdict = {"exercise": exercise["name"], "type": exercise["type"], "events": []}

    if exercise["type"] == "riff":
        truth = exercise["events"]
        pairs, extras = _match_notes(truth, analysis["notes"], tol_ms, tol_cents)
        for p in pairs:
            ev = {"label": p["event"].get("label", ""), "expected_midi": p["event"].get("midi"),
                  "expected_name": midi_name(p["event"]["midi"])}
            if p["match"] == "none":
                ev["status"] = "missed"
                ev["hint"] = "not heard at all — check the gain or count it out loud"
            else:
                ev["dt_ms"] = round(p["dt_ms"])
                ev["cents"] = round(p["cents"])
                pitch_bad = abs(p["cents"]) > tol_cents
                time_bad = abs(p["dt_ms"]) > tol_ms
                if p["match"] == "lenient" and pitch_bad:
                    played = p.get("played_midi")
                    interval = p["event"]["midi"] - played if played else 0
                    if abs(interval) in (1, 2):
                        ev["status"] = "wrong_fret"
                        ev["hint"] = (f"played {midi_name(played)}, wanted {ev['expected_name']} — "
                                      f"{'one fret too low' if interval > 0 else 'one fret too high'}")
                    elif abs(interval) <= 12:
                        ev["status"] = "wrong_note"
                        ev["hint"] = f"played {midi_name(played)}, wanted {ev['expected_name']} — wrong string/fret"
                    else:
                        ev["status"] = "wrong_note"
                        ev["hint"] = f"played {midi_name(played)}, wanted {ev['expected_name']}"
                elif time_bad:
                    ev["status"] = "timing"
                    ev["hint"] = "late — behind the beat" if p["dt_ms"] > 0 else "early — rushing"
                elif pitch_bad:
                    ev["status"] = "pitch"
                    ev["hint"] = ("sharp — ease fret pressure" if p["cents"] > 0
                                  else "flat — fret closer to the fretwire")
                else:
                    ev["status"] = "ok"
            verdict["events"].append(ev)
        if extras:
            verdict["extra_notes"] = [
                {"midi": int(round(e["midi"])), "name": midi_name(int(round(e["midi"]))),
                 "t": round(e["t0"], 2), "hint": "unplanned note — string noise or missed mute"}
                for e in extras]
        n = len(pairs)
        ok = sum(1 for e in verdict["events"] if e["status"] == "ok")
        verdict["score"] = round(100 * ok / n) if n else 0

    elif exercise["type"] == "chords":
        pairs = _match_chords(exercise["events"], analysis["chords"], tol_ms)
        for p in pairs:
            ev = {"label": p["event"].get("label", ""), "expected": p["event"]["chroma"]}
            if p.get("missed"):
                ev["status"] = "missed"
            else:
                cov = p["coverage"]
                ev["coverage"] = round(cov, 2)
                if cov > 0.92:
                    ev["status"] = "ok"
                elif cov > 0.80:
                    ev["status"] = "muddy"
                    ev["hint"] = _chord_hint(p["tones"], p["chroma"]) or "a bit muddy — let each string ring"
                else:
                    ev["status"] = "wrong"
                    ev["hint"] = _chord_hint(p["tones"], p["chroma"]) or "that's not the target chord"
            verdict["events"].append(ev)
        n = len(pairs)
        ok = sum(1 for p in pairs if not p.get("missed") and p.get("coverage", 0) > 0.92)
        verdict["score"] = round(100 * ok / n) if n else 0

    # global tempo verdict from median dt
    dts = [e["dt_ms"] for e in verdict["events"] if "dt_ms" in e]
    if len(dts) >= 3:
        med = float(np.median(dts))
        verdict["tempo_drift_ms"] = round(med)
        if abs(med) > tol_ms:
            verdict["tempo_hint"] = ("you consistently drag — try 80% speed with a metronome"
                                     if med > 0 else "you rush — relax the picking hand")
    return verdict


def _worst_pc(target, detected):
    """Name the pitch class that should sound but is weakest (or rings wrongly)."""
    worst, worst_val = None, 1e9
    for name, want in target.items():
        if want >= 0.8:  # must-sound notes only
            got = detected.get(name, 0)
            if got < worst_val:
                worst, worst_val = name, got
    if worst and worst_val < 0.12:
        return f"'{worst}' isn't sounding — that string is muted or not fretted"
    # unexpected loud pitch class
    loudest, loudest_val = None, 0
    for name, want in target.items():
        if want < 0.2:
            got = detected.get(name, 0)
            if got > loudest_val:
                loudest, loudest_val = name, got
    if loudest and loudest_val > 0.2:
        return f"'{loudest}' rings when it shouldn't — check your fingering"
    return None


def summarize_for_cloud(verdicts):
    """Compress one or more verdicts into the text a cloud LLM should see."""
    lines = [f"Exercise: {verdicts[0]['exercise'] if verdicts else '?'}"]
    for v in verdicts:
        lines.append(f"pass score: {v.get('score')}%")
        for e in v["events"]:
            if e["status"] == "ok":
                continue
            desc = f"  - {e.get('label','?')}: {e['status']}"
            if "dt_ms" in e:
                desc += f" ({e['dt_ms']:+d} ms)"
            if "hint" in e:
                desc += f" — {e['hint']}"
            lines.append(desc)
        for x in v.get("extra_notes", []):
            lines.append(f"  - extra note {x['name']} at {x['t']}s — {x['hint']}")
        if "tempo_hint" in v:
            lines.append(f"  - tempo: {v['tempo_hint']}")
    return "\n".join(lines)
