# Voices

Reference voices used for cloning, grouped by the voice-design engine that produced them:
`voices/<designer>/<voice>/`. Experiments refer to a voice by that path, e.g.
`voice: qwen3-tts-1.7b-voicedesign/warm_male`, so the same voice described to a different designer can be cloned and
compared side by side. Each voice folder holds:

| File | Content |
|---|---|
| `reference.wav` | ~10 s reference clip |
| `transcript.txt` | exact transcript of the clip (cloning needs it) |
| `description.txt` | the text description it was designed from |
| `metadata.json` | designer, seed and scores of the chosen take |

All of them are **synthetic**: designed from a text description in [experiment 01](../experiments/01-voice-design).
No real person's voice is cloned in this repository. Re-running experiment 01 regenerates them; the copies committed
here are the ones the published results were produced with.

| Voice | Designer | Use |
|---|---|---|
| `warm_male` | `qwen3-tts-1.7b-voicedesign` | the main voice of every experiment: calm, warm, mid-forties man |
| `warm_male_energetic` | `qwen3-tts-1.7b-voicedesign` | the same man described as euphoric and high-pitched; the "bad reference" in experiment 04 |

To add a designer, register a voice-design engine (`supports_voice_design = True`, see
[`labtts/engines/base.py`](../labtts/engines/base.py)) and add it under `engines:` in experiment 01's
[`config.yaml`](../experiments/01-voice-design/config.yaml): it designs the same voices from the same descriptions.
