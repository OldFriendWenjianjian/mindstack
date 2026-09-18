# MindStack · 心智栈

**English** | [中文](#中文)

A general-purpose AI companion architecture: one small model on your own
GPU turns camera / microphone / motor-joint / any-sensor signals into a
compact "feature soup", the cloud big model (GLM-5.3-Flash) reads the
soup and sends back feedback as actions, and hardwired reflexes keep
everything safe with zero model involvement.

One loop. Any device. Sensors are the only interface.

---

## 中文

一套通用 AI 伙伴架构：本地 GPU 上的小模型把摄像头、麦克风、电机关节、
任何传感器的原始信号提炼成一份"特征汤"（feature soup），云端大模型
（GLM-5.3-Flash）读取特征汤、以 JSON 动作指令的形式反馈回来，
而硬编码的"脊髓反射"负责安全兜底——完全不需要任何模型参与。

一套循环，接任何设备。传感器就是唯一的交互界面。

## The architecture · 架构

```
 camera(s) ─┐
 mic(s) ────┤   ┌──────────────┐   feature    ┌───────────────────┐
 joints ────┼──>│ LOCAL MODEL  │──> soup ───> │ CLOUD GLM-5.3-Flash│
 sensors ───┘   │ (GPU, small) │   (JSON,     │  feedback: thought │
                └──────────────┘    cheap)    │  say + actions[]   │
                       ▲                      └─────────┬─────────┘
                       │                                │ actions
                ┌──────┴───────┐                                │
                │ REFLEXES     │  veto / override, <1ms         │
                │ (spinal cord)│<───────────────────────────────┘
                └──────┬───────┘
                       ▼
                 actuators (amp, motor, video-gen, ...)
```

| Layer · 层 | What · 职责 | Model? · 用模型吗 |
|---|---|---|
| Reflexes · 脊髓反射 | hardwired safety: stall / overcurrent / howl / VRAM watch → immediate action. Pure functions, work offline · 硬编码安全：堵转/过流/啸叫/显禁看门狗，纯函数、离线可用 | never · 永不 |
| Soup · 特征汤 | GPU feature extraction: spectra, motion energy, telemetry stats — bounded tokens regardless of input length · GPU 特征提取：频谱、运动能量、遥测统计，令牌数恒定 | small local only · 仅本地小模型 |
| Mind · 大脑 | read soup → judge → `{"thought", "say", "actions"}`; strict JSON contract, violations dropped · 读汤→判断→严格 JSON 契约，违规即弃 | cloud GLM · 云端 |
| Memory · 记忆 | session JSONL logs; recent history re-enters context each session · 会话日志，历史重入上下文 | free · 免费 |

### Safety boundary · 安全边界

The deliberator is **advisory-only, structurally**: reflexes are pure
functions with actuation authority; the LLM can propose, reflexes
dispose. `classify(sample) -> action` would keep the robot safe with
every network and model dead. · 大模型在结构上只有建议权：反射是拥有
执行权的纯函数。断网断模型，机器人依然安全。

## Do I need to train it? · 需要训练吗？

**No, to start.** The cloud model already "went to school" (pretraining);
the soup is hand-written math; reflexes are deliberately unlearned.
Teaching it is like teaching a kid, cheapest method first:

| Teach a child · 教孩子 | Teach this system · 教这套系统 | Cost · 成本 |
|---|---|---|
| Talk to him · 说话引导 | system prompt（角色、规则）| free · 免费 |
| Demonstrate · 示范 | examples in the prompt（好的/坏的反馈示例）| free · 免费 |
| He remembers · 他会记得 | session logs re-read as memory · 会话日志即记忆 | free · 免费 |
| Habits over months · 养成习惯 | retrieval over logs · 日志检索 | small build · 小改动 |
| School · 上学 | QLoRA fine-tune a local model on *your* videos + teacher-style labels（用你的录像+教师评语微调本地模型，几百段即可）| hours · 数小时 |

Key difference from a baby: an LLM **learns inside a conversation and
forgets at its end** — day-to-day "teaching" is prompting + logs.
Fine-tune only when a specific repeated failure survives good prompts.
· 与婴儿的关键区别：大模型在对话内学习、对话结束即遗忘——日常"教"
就是写好提示词+攒日志。只有当好提示词也治不了某个反复出现的毛病时，
才值得微调。

## Run · 运行

```bash
pip3 install numpy scipy soundfile pytest opencv-python-headless pillow requests torch

# cloud key: export GLM_API_KEY=...   (or auto-read from ZCode config · 自动读取)
# local assist (optional · 可选):
#   export LOCAL_LLM_URL=http://127.0.0.1:11434/v1   # Ollama
#   ollama pull qwen3:14b        # fits a 16GB GPU · 16GB 显存即可

# tests · 测试
python3 -m pytest tests/ -q
```

### One loop, any device · 一套循环，接任何设备

```python
from mind.loop import Loop
from adapters import AmpAdapter, AmpReflex      # or JointAdapter, VideoGenAdapter

loop = Loop(AmpAdapter(mic_source), AmpReflex(),
            "You are my amp assistant. Watch levels, protect the speakers.")
loop.cycle(question="How does the amp look?")
```

A new device = one adapter (`read()` + `act()`, ~30 lines) + a reflex
policy + a role prompt. No new app. · 接一个新设备 = 写一个适配器
（read + act，约 30 行）+ 一条反射策略 + 一段角色提示词。不需要新应用。

## Status · 状态

Verified: soup extraction (audio/video/telemetry), loop mechanics with
reflex veto (20 tests), joint reflexes against the real STM32 firmware
telemetry format, cloud feedback round-trips. Live mic / real camera /
real serial bring-up happens on the target PC — this build box has none
attached.

已验证：特征汤提取（音频/视频/遥测）、含反射否决的循环机制（20 项测试）、
关节反射对齐真实 STM32 固件遥测格式、云端反馈往返。实时麦克风/真实摄像头/
真实串口联调需在目标机器上进行——当前构建机未接这些硬件。

## Honest limits · 如实说明

- The cloud model advises; it cannot act on hardware directly. This is
  by design. · 云端模型只提供建议，不能直接操作硬件——这是设计使然。
- Reflex thresholds are hand-tuned defaults; tune them to your hardware
  before trusting them. · 反射阈值是手工默认值，请按你的硬件调整后再信任。
- The soup is classic DSP/CV. Swapping in a learned encoder later is an
  upgrade path, not a requirement. · 特征汤目前是经典 DSP/CV，日后可换
  学习型编码器，但并非必需。
