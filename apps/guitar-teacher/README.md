# guitar-teacher

A real-time guitar teacher for a local machine, built on the brain-layer
architecture: cheap deterministic DSP reacts at audio rate, rules grade
your pass instantly, an optional local LLM softens the wording, and the
cloud model (GLM) acts as the actual teacher — planning, explaining,
prescribing drills. The cloud never sees audio, only the coach's
structured verdict, so tokens stay cheap and latency stays bounded.

```
 mic ──> DSP (reflex)   ──> Coach (rules)  ──> you, instantly
        YIN/chroma/flux      grade + hints         │
                             │ summary             ▼
                             └──────────> [local LLM phrasing, optional]
                                          [cloud GLM teacher] ──> drills
```

## Status

Working and verified on synthetic takes with ground truth (see
`tests/`): riff pitch/timing grading, wrong-fret diagnosis ("played F4,
wanted G4 — one fret too low"), tempo-drift detection ("you consistently
drag"), chord coverage grading with per-string hints. Live mic mode and
the local-LLM hook are implemented but were only smoke-tested here —
this box has no mic or GPU; first-run tuning on your machine is expected.

## Install (Pop!_OS, 3900X + 4060 Ti 16GB)

```bash
sudo apt install -y python3-pip portaudio19-dev
pip3 install numpy scipy soundfile pytest sounddevice   # use a venv if you prefer

# optional: local phrasing model (fast, fits 16GB VRAM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:14b        # ~9 GB, fully in VRAM
```

Ollama exposes an OpenAI-compatible API on port 11434. Point the phrasing
layer at it and enable it per-run:

```bash
export LOCAL_LLM_URL=http://127.0.0.1:11434/v1
export LOCAL_LLM_MODEL=qwen3:14b
```

Cloud teacher uses the same GLM endpoint ZCode uses (key read from
`/root/.zcode/cli/config.json`, or `export GLM_API_KEY=...`).

## Use

```bash
# grade one recorded take (from any recorder, e.g. arecord / Audacity)
python3 teacher.py pass exercises/riff_schema.json my_take.wav --speak

# coach live: plays are graded pass after pass, Ctrl-C to stop
python3 teacher.py live exercises/riff_schema.json

# talk to the cloud teacher; it remembers your recent passes
python3 teacher.py ask "my G always buzzes on the low E string — why?"

# run the full synthetic demo
python3 teacher.py demo
```

`--speak` routes the top fault lines through the local model for a
friendlier spoken tone; without a local model the rule-based hints are
used as-is and nothing else changes.

## Exercises

JSON files in `exercises/`. Two types:

- `riff` — note events with midi, time, duration; graded on pitch
  (cents), timing (ms), and wrong-note diagnosis.
- `chords` — strummed chords graded by chroma coverage: what fraction of
  the heard energy sits on chord tones, with hints naming the string
  that's muted or the open string ringing that shouldn't.

Write your own by copying the schema; the synth in `make_test_audio.py`
renders any exercise to audio, which doubles as a ground-truth generator.

## Layout

| path | layer | role |
|------|-------|------|
| `dsp/analysis.py` | reflex | YIN pitch, spectral-flux onsets, chroma, note tracking |
| `coach/grade.py` | rules | exercise diff, verdicts, fault hints, cloud summary |
| `coach/local_phrase.py` | local LLM | optional rephrasing, never blocks |
| `cloud/teacher.py` | deliberation | GLM client, session history |
| `teacher.py` | CLI | pass / live / ask / demo |
| `session.jsonl` | memory | every pass + every cloud reply, JSONL |

## Honest limits

- Monophonic pitch only for riffs: single-string practice. Chords are
  judged by chroma, not per-string polyphony.
- Tuning reference assumed A440; capo or drop tunings need exercise
  transposition.
- Live latency: a pass is analyzed after you finish it (not
  score-following in real time). Sub-second "you're dragging right now"
  feedback would need a beat-tracker — next step if wanted.
- First runs on real guitar (not synth) may need tolerance retuning
  (`tolerance_cents`, `tolerance_ms` in the exercise JSON) — expect
  30–50 cents of real-guitar pitch scatter at the 12th fret and below.
