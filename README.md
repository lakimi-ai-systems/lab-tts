# lab-tts

Reproducible text-to-speech experiments in **Castilian Spanish**, starting with
[Qwen3-TTS 0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).

Every experiment answers one question, runs with one command and regenerates all of its audio: no audio is stored in
the repository except the short synthetic reference voices in [`voices/`](voices), and the published numbers live
in the experiment READMEs. The benchmarks are written against a common engine interface, so new models can be added
and compared on the same texts, voices and judges.

| Engine | Model | Can |
|---|---|---|
| `qwen3-tts-0.6b` | [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | clone |
| `audio8-air` | [Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) | clone, or use its built-in voice |
| `qwen3-tts-1.7b-voicedesign` | [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | design voices from a description |
| `qwen3-tts-1.7b` | [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | clone (registered, not benchmarked yet) |

## Findings

Summaries of each experiment's conclusions; the full tables are in the experiment READMEs.

| # | Question | Short answer |
|---|---|---|
| [01](experiments/01-voice-design) | Can we design a good reference voice without recording anyone? | **Yes** (Qwen3-TTS 1.7B VoiceDesign). All calm takes are fully intelligible. Generate several: the same description gives voices from 94 to 199 Hz. The English description gave calmer takes than the Spanish one. |
| [02](experiments/02-phonetics) | How well does each engine pronounce Castilian Spanish? | **Qwen 0.6B: very well.** 8 of 11 hard texts flawless; the weak spot is non-Castilian spellings such as *Sanxenxo*, and lower temperature does not help. **Audio8: good, a step behind** (phonetic error 0.0099 vs 0.0069); it drops a syllable in *electroencefalograma* every time. |
| [03](experiments/03-long-form) | Two minutes of speech: one shot or sentence by sentence? | **Qwen: both hold up** (WER 0). One shot flows better but fades 3–5 dB; sentence by sentence stays level. Feed several sentences per call. **Audio8: sentence by sentence only**; in one go it stops after 2–3 s. |
| [04](experiments/04-shrill-voice) | Why does a cloned voice sometimes turn shrill? (Qwen 0.6B) | **Mostly the reference**, which the clone copies faithfully. **Occasional** shrill moments come from a hot talker temperature (1.2), not from the subtalker. Re-cloning from generated audio makes the voice duller, not shrill. |
| [05](experiments/05-everyday-sentence) | One everyday sentence (the samples posted on X): how do the engines compare? | **All read it correctly.** With the same cloned voice, **Audio8 spends twice as long pausing** (5.2 s vs 2.6 s, takes 19 s vs 15 s): that is its "out of breath" feel. |
| [06](experiments/06-max-text-length) | How much text fits in one generation? | **Qwen 0.6B: the full 375 words** (129 s). **Audio8: about 70 words** (~25 s); beyond that it derails or stops, without running out of context. |

## The recipe

Design a voice with the 1.7B VoiceDesign model, then clone it with the small 0.6B Base model:

```python
import torch
from qwen_tts import Qwen3TTSModel

options = dict(device_map="cuda:0", dtype=torch.bfloat16)
reference_text = "Hola, buenas tardes, gracias por llamar. Cuénteme con calma qué necesita..."

# 1. Describe the voice. Generate several takes and keep the calmest clean one (~10 s)
designer = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", **options)
waves, sample_rate = designer.generate_voice_design(
    text=reference_text, language="Spanish",
    instruct="A man in his mid forties with a warm, friendly, medium-deep voice. Calm, unhurried pace. "
             "Native Castilian Spanish accent from Spain.")

# 2. Build the clone prompt ONCE, with the exact transcript, and reuse it for every generation
cloner = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", **options)
prompt = cloner.create_voice_clone_prompt(ref_audio=(waves[0], sample_rate), ref_text=reference_text)
waves, sample_rate = cloner.generate_voice_clone(text="Cualquier texto nuevo.", language="Spanish",
                                                 voice_clone_prompt=prompt, max_new_tokens=2048)
```

[Experiment 01](experiments/01-voice-design/run.py) automates the take selection, and
[`labtts/engines/qwen3_tts.py`](labtts/engines/qwen3_tts.py) wraps the cloning side.

## Getting started

Requirements: Linux (or WSL2), an NVIDIA GPU with ~10 GB of free memory, [uv](https://docs.astral.sh/uv/) and
`ffmpeg` (only for the video tool).

```bash
uv sync                                               # pinned environment (torch 2.8 + CUDA 12.8)
uv run python experiments/01-voice-design/run.py      # designs the reference voices
uv run python experiments/02-phonetics/run.py         # ...and so on, or all at once:
tools/run_all.sh
```

Everything an experiment produces goes to `results/<experiment>/`, which is git-ignored: the audio, under
`<engine>/`, and `results.json`, with every take's scores and one entry per engine with the exact environment it was
produced in (package versions, GPU, git commit). Each `run.py` then refreshes the results tables in its README, which
is what the repository keeps. Two options are shared by every experiment:

- `--engines qwen3-tts-0.6b,other` runs only those engines and keeps the results of the others from your earlier runs.
- `--rescore` judges the audio already generated again, without regenerating it (after changing a judge or metric).

## How speech is judged

- **Whisper WER** ([`labtts/judges/whisper.py`](labtts/judges/whisper.py)). The usual metric. Good at catching
  skipped or invented words, but its language model silently fixes mispronunciations, so a take can score a perfect
  WER and still sound wrong.
- **Phonetic judge** ([`labtts/judges/phonetic.py`](labtts/judges/phonetic.py)). A character-level wav2vec2 CTC
  recognizer for Spanish decoded greedily, with no language model, so it writes down what actually sounds. Text and
  transcript are mapped to an approximate Peninsular Spanish phonetic spelling (b/v or a silent h are not errors) and
  aligned word by word. Reported as the share of edited characters ("phonetic error") plus the flagged words.
- **Acoustic metrics** ([`labtts/metrics.py`](labtts/metrics.py)). Pitch (pYIN), loudness, brightness (spectral
  centroid of voiced frames) and speaking rate, globally or over time windows.

None of these replaces listening. They find the takes worth listening to.

## Repository layout

```
labtts/
  engines/       one module per TTS model behind a common interface, plus the registry of engine names
  judges/        Whisper WER and the phonetic judge
  metrics.py     pitch, loudness, brightness, speaking rate, shrillness
  voices.py, texts.py, records.py, paths.py
texts/           test texts (neutral, written to stress known difficulties of Spanish)
voices/          synthetic reference voices from experiment 01: voices/<designer>/<voice>/
experiments/     one folder per experiment: README, config.yaml, run.py
results/         generated audio and results.json per experiment (git-ignored)
tools/           run_all.sh, waveform_video.sh (waveform video for sharing a sample)
```

## Adding an engine

1. Write `labtts/engines/<model>.py` with a subclass of [`Engine`](labtts/engines/base.py): load the model in
   `__init__`, implement `synthesize(text, seed, settings)` (one generation of the whole text, settings being the
   variant's overrides on top of your defaults) and `close()`. Set `supports_cloning = True` and implement
   `use_reference()` if the model clones; otherwise it speaks with its built-in voice. A model that designs voices
   from a description sets `supports_voice_design = True` and implements `design_voice()`; added to experiment 01,
   it designs the same voices into `voices/<its name>/`, ready to be cloned by any engine.
2. Register a name for it in [`labtts/engines/__init__.py`](labtts/engines/__init__.py).
3. If it needs extra packages, add them as an optional dependency group in `pyproject.toml` (`uv sync --extra <name>`).
   If they conflict with the core environment, generate in a separate environment and judge with `--rescore` here.
4. Add the engine and its variants under `engines:` in the `config.yaml` of the benchmarks it should run, then run
   them with `--engines <name>`: the other engines' results are kept.

Behaviour specific to one model (say, its own sampling knobs or artefacts) deserves its own experiment rather than
a column in a shared benchmark.

## Adding an experiment

Copy [`experiments/TEMPLATE.md`](experiments/TEMPLATE.md) into a new `NN-name/` folder, with a `config.yaml` and a
`run.py` following the existing ones.

## Responsible use

The voices here are designed from text descriptions, not cloned from real people. If you clone a real voice, get the
speaker's consent.

## License

Code under [Apache 2.0](LICENSE). The models are subject to their own licenses (Qwen3-TTS and Audio8-TTS-Realistic-Air: Apache 2.0).
