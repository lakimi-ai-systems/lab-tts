# Voices

Reference voices used for cloning. Each folder holds:

| File | Content |
|---|---|
| `reference.wav` | ~10 s reference clip |
| `transcript.txt` | exact transcript of the clip (the clone prompt needs it) |
| `description.txt` | the text description it was designed from |
| `metadata.json` | model, seed and scores of the chosen take |

All of them are **synthetic**: designed from a text description with Qwen3-TTS VoiceDesign in
[experiment 01](../experiments/01-voice-design). No real person's voice is cloned in this repository. Re-running
experiment 01 regenerates them; the copies committed here are the ones the published results were produced with.

| Voice | Use |
|---|---|
| `warm_male` | the main voice of every experiment: calm, warm, mid-forties man |
| `warm_male_energetic` | the same man described as euphoric and high-pitched; the "bad reference" in experiment 04 |
