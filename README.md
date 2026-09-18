# MindStack

A layered robot-mind architecture: hardwired reflexes at the bottom,
cheap deterministic senses above them, and LLM deliberation at the top —
local when it's small and fast, cloud (GLM) when it's deep. One shared
architecture, several "apps" that are really just different senses
plugged into the same stack:

- **Guitar teacher** — listens to you play, grades pitch/timing/chords
  instantly, and the cloud teacher prescribes drills.
- **Movie companion** — watches a film with you, follows the shot
  structure, describes storyline and emotional arc.
- **Cat companion** — watches a kitten-cam, detects her active moments,
  reads her mood from frames, and stays silent (and free) while she sleeps.
- **Joint guardian** — supervises FOC motor joints (STM32 + real motor
  firmware): hardwired safety reflexes with zero model involvement, and
  an advisory cloud analyst that can never touch the controls.

## The architecture

The design copies the layering of the human nervous system: never spend
a slow, expensive layer on anything a faster, cheaper layer can own.

```
                 ┌─────────────────────────────────────────────┐
  deliberate     │  CLOUD (GLM, slow ~1-30s, sees summaries)   │  teacher,
  (cortex)       │  storyline · drills · trends · advisory     │  analyst
                 └───────────────▲─────────────────────────────┘
                                 │ distilled facts only, never raw streams
                 ┌───────────────┴─────────────────────────────┐
  assist         │  LOCAL LLM (1-14B on your GPU, <1s)         │  optional
  (System 1)     │  rephrasing · triage · routing              │
                 └───────────────▲─────────────────────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────┐
        │               ┌────────┴────────┐                │
  sense          │  SENSES (deterministic DSP/CV, ms, no model)│  salience /
  (brainstem)    │  YIN pitch · chroma · spectral flux         │  colliculus
                 │  motion energy · scene cuts · region gaze   │
                 └───────────────▲─────────────────────────────┘
                                 │ events + features only
                 ┌───────────────┴─────────────────────────────┐
  reflex         │  REFLEX POLICY (pure functions, <1ms)       │  RESPONDS:
  (spinal cord)  │  current/heat/stall/bus limits ->           │  pwm_off,
                 │  immediate safe actions, no network needed  │  iq_zero
                 └─────────────────────────────────────────────┘
```

### The rules that make it work

1. **Data moves up only as distilled summaries.** The deliberator never
   sees raw audio, frame streams, or 4 Hz telemetry — it sees the
   coach's verdict list, the shot descriptions, the 30-second telemetry
   digest. Tokens stay bounded no matter how long the session runs.
2. **Events move down only as gain.** Higher layers don't take over
   lower ones; they bias them (like top-down attention): which region to
   grab frames from, which faults matter, whether to escalate at all.
3. **Reflexes are non-negotiable.** Safety actions come from pure
   functions of the current sensor sample. No model call, no network,
   no queue. `classify(sample) -> action` is the whole contract, and it
   would keep the robot safe with every LLM on Earth offline.
4. **The deliberator has zero actuation authority.** It advises; humans
   and reflexes act. This is the safety boundary, and it's structural —
   not a promise in a prompt.
5. **Attention gates the expensive path.** A sleeping kitten, a silent
   guitar, a healthy motor all produce *no model calls*. Salience
   detectors (motion bursts, faults, stalls) are the only things that
   wake the mind. This mirrors the brain's own economy and it's why a
   24/7 companion is affordable.

### Why not just "an app that calls GPT"?

Because latency, cost, and safety have structure:

| concern | naive LLM app | mindstack |
|---|---|---|
| a stalling motor (50 ms to act) | round-trip to an LLM | reflex function, <1 ms |
| watching video for 8 h | per-frame calls: $$$$ | per-salient-event: cents |
| "why does my G buzz?" | generic chat | coach facts + session history in context |
| movie storyline | send 1000 frames | 12 shot descriptions |
| local/offline mode | none | senses+reflexes keep working; local LLM optional |

## Repository layout

```
mindstack/
├── senses/          deterministic perception, no models
│   └── vision.py      VideoScan: motion energy, scene cuts, segments,
│                      region-of-motion gaze, JPEG keyframe export
├── reflex/          (per-app hardwired policies live with their app;
│   └── ...            joint_app.classify is the reference example)
├── mind/
│   └── client.py    one LLM interface: local routing, cloud GLM,
│                     vision frames, reasoning-budget retry
├── apps/
│   ├── guitar-teacher/   hearing sense + coach + teacher (full app, tests)
│   ├── movie_app.py      video -> shots -> storyline/emotion
│   ├── cat_app.py        kitten-cam: motion bursts -> mood readings
│   ├── joint_app.py      FOC joint telemetry -> reflex actions + advisor
│   ├── make_test_movie.py / make_test_cat.py   ground-truth generators
│   └── guitar-teacher/README.md                app-specific setup
└── tests/           reflex/parser unit tests (real firmware line format)
```

## The FOC connection

`joint_app.py` speaks the real telemetry protocol of the STM32G473
UE6815 FOC firmware that lives in this workspace
(`st= ang= pwm= vbus= id= iq= vd= vq= mod= ...` at 4 Hz over USART1,
115200 8N1). The reflex limits are derived from that motor's actual
numbers: software iq clamp 8 A (reflex trips at 6 A), field-weakening
edge −10 A, SVPWM modulation ceiling 0.97, 48 V nominal bus. A
supervisor process opens the CH340 serial port, feeds lines to
`JointSupervisor.feed()`, and executes the returned actions
(`pwm_off` / `iq_zero`) as serial commands — fast enough that a stalled
or overheating joint is cut before damage, with the cloud analysis
arriving seconds later to explain why.

Extending to tactile/heat sensors is deliberately boring: their ADC
channels appear in telemetry tokens, `classify()` gains two limit
checks, done. The spinal cord doesn't need a model upgrade.

## Running

```bash
pip3 install numpy scipy soundfile pytest opencv-python-headless pillow requests

# cloud key (or export GLM_API_KEY=...)
# key is auto-read from ZCode config if present

# optional local assist layer:
#   export LOCAL_LLM_URL=http://127.0.0.1:11434/v1   (Ollama)
#   export LOCAL_LLM_MODEL=qwen3:14b

# guitar
python3 apps/guitar-teacher/teacher.py demo
python3 apps/guitar-teacher/teacher.py live apps/guitar-teacher/exercises/riff_schema.json

# movie companion
python3 apps/movie_app.py my_movie.mp4

# cat companion (file or live camera)
python3 apps/cat_app.py kitten_clip.mp4
python3 apps/cat_app.py /dev/video0 --live

# tests
python3 -m pytest tests/ apps/guitar-teacher/tests -q
```

## Status & honest limits

Verified on this machine, end to end: guitar grading (7 tests, all
fault classes), movie scan on a ground-truth synthetic film (cuts at
4.0/8.0 s found exactly; coherent storyline back from GLM), cat window
detection + mood reading, joint reflex policy (10 tests) + real cloud
consult. **Not yet verified on real hardware**: live mic guitar input,
a real kitten-cam, and the serial-link joint supervisor — the code paths
are implemented but were exercised with synthetic signals only, because
this build box has no microphone, camera, or motor attached.

Local-LLM layers (`qwen3:14b` via Ollama fits a 16 GB GPU) are optional
everywhere; without them everything still works, just with rule-based
wording and cloud-only deliberation.

## Adding a new sense

1. Write the deterministic feature extractor in `senses/` (pure
   numpy/cv2, milliseconds, no network).
2. Write the salience rule: what event makes this sense worth a model
   call?
3. Write the distillation: what 10-line summary does the deliberator
   need?
4. Only then touch `mind/client.py` — usually not at all.

If a safety action exists, it belongs in a reflex function, not in a
prompt.
