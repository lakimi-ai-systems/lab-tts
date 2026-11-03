# lab-tts

Reproducible text-to-speech experiments in **Castilian Spanish**, starting with
[Qwen3-TTS 0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).

Every experiment answers one question, runs with one command and regenerates all of its audio: no audio is stored in
the repository except the short synthetic reference voices in [`voices/`](voices). The benchmarks are written
against a common engine interface, so new models can be added and compared on the same texts, voices and judges.

| Engine | Model | Cloning |
|---|---|---|
| `qwen3-tts-0.6b` | [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | yes |
| `qwen3-tts-1.7b` | [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) (registered, not benchmarked yet) | yes |

## Findings

So far for `qwen3-tts-0.6b`. Summaries of each experiment's conclusions; the full tables are in the experiment
READMEs.

| # | Question | Short answer |
|---|---|---|
| [01](experiments/01-voice-design) | Can we design a good reference voice without recording anyone? | **Yes.** All calm takes are fully intelligible. Generate several: the same description gives voices from 94 to 199 Hz. The English description gave calmer takes than the Spanish one. |
| [02](experiments/02-phonetics) | How well does the 0.6B model pronounce Castilian Spanish? | **Very well.** 8 of 11 hard texts are flawless (z/c, jota, rr, ñ, numbers, questions...). The weak spot is non-Castilian spellings such as *Sanxenxo*. Lower temperature does not help. |
| [03](experiments/03-long-form) | Two minutes of speech: one shot or sentence by sentence? | **Both hold up** (WER 0). One shot flows better but fades 3–5 dB over two minutes; sentence by sentence stays level. Feed several sentences per call. |
| [04](experiments/04-shrill-voice) | Why does a cloned voice sometimes turn shrill? | **Mostly the reference**, which the clone copies faithfully. **Occasional** shrill moments come from a hot talker temperature (1.2), not from the subtalker. Re-cloning from generated audio makes the voice duller, not shrill. |

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

Audio is written to `results/<experiment>/<engine>/` (git-ignored). Each `run.py` also writes `results.json` next to
it, with one entry per engine and the exact environment each was produced in (package versions, GPU, git commit), and
refreshes the results tables in its README. Two options are shared by the engine benchmarks (02–04):

- `--engines qwen3-tts-0.6b,other` runs only those engines and keeps the stored results of the rest.
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
voices/          synthetic reference voices produced by experiment 01
experiments/     one folder per experiment: README, config.yaml, run.py, results.json
tools/           run_all.sh, waveform_video.sh (waveform video for sharing a sample)
```

## Adding an engine

1. Write `labtts/engines/<model>.py` with a subclass of [`Engine`](labtts/engines/base.py): load the model in
   `__init__`, implement `synthesize(text, seed, settings)` (one generation of the whole text, settings being the
   variant's overrides on top of your defaults) and `close()`. Set `supports_cloning = True` and implement
   `use_reference()` if the model clones; otherwise it speaks with its built-in voice.
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

Code under [Apache 2.0](LICENSE). The models are subject to their own licenses (Qwen3-TTS: Apache 2.0).
