"""Experiment 01: design a voice from a text description and pick the reference take automatically.

1. Qwen3-TTS 1.7B VoiceDesign says the same ~10 s sentence for every description and seed.
2. Each take is scored: Whisper WER, phonetic judge, pitch profile.
3. Among the takes that are fully intelligible, the calmest (lowest pitch spread) becomes voices/warm_male and the
   liveliest of the energetic description becomes voices/warm_male_energetic.

Usage: uv run python experiments/01-voice-design/run.py
"""
from pathlib import Path

import soundfile as sf
import torch

from labtts.engines.qwen3_tts import design_voice, load_model
from labtts.judges.phonetic import PhoneticJudge
from labtts.judges.whisper import WhisperJudge, word_error_rate
from labtts.metrics import analyze_frames, voice_profile
from labtts.paths import experiment_output_dir
from labtts.records import markdown_table, update_readme_results, write_results
from labtts.texts import load_config
from labtts.voices import save_voice

EXPERIMENT_DIR = Path(__file__).resolve().parent


def generate_candidates(config: dict, output_dir: Path) -> list[dict]:
    model = load_model(config["model"])
    candidates = []
    for voice_name, voice in config["voices"].items():
        for description_language, description in voice["descriptions"].items():
            for seed in config["seeds"]:
                synthesis = design_voice(model, config["reference_text"], description, config["sampling"], seed)
                path = output_dir / f"{voice_name}_{description_language}_seed{seed}.wav"
                sf.write(path, synthesis.wave, synthesis.sample_rate)
                candidates.append({"voice": voice_name, "description_language": description_language, "seed": seed,
                                   "wav": path.name, "duration_s": round(synthesis.duration_seconds, 2),
                                   "real_time_factor": round(synthesis.real_time_factor, 3)})
                print(f"generated {path.name}: {synthesis.duration_seconds:.1f} s", flush=True)
    del model
    torch.cuda.empty_cache()
    return candidates


def score_candidates(config: dict, candidates: list[dict], output_dir: Path) -> None:
    phonetic_judge, whisper_judge = PhoneticJudge(), WhisperJudge()
    for candidate in candidates:
        wave, sample_rate = sf.read(output_dir / candidate["wav"], dtype="float32")
        verdict = phonetic_judge.evaluate(config["reference_text"], wave, sample_rate)
        transcript = whisper_judge.transcribe(wave, sample_rate)
        profile = voice_profile(analyze_frames(wave, sample_rate))
        candidate.update({"phonetic_error_rate": round(verdict.error_rate, 4), "flagged_words": verdict.flagged_words,
                          "word_error_rate": round(word_error_rate(config["reference_text"], transcript.text), 4),
                          "whisper_transcript": transcript.text, "profile": profile.as_dict()})


def pick_references(config: dict, candidates: list[dict], output_dir: Path) -> dict:
    limits = config["selection"]
    chosen = {}
    for voice_name, voice in config["voices"].items():
        eligible = [candidate for candidate in candidates if candidate["voice"] == voice_name
                    and candidate["word_error_rate"] <= limits["max_word_error_rate"]
                    and len(candidate["flagged_words"]) <= limits["max_flagged_words"]]
        if not eligible:
            raise RuntimeError(f"No intelligible take for {voice_name}: add seeds or relax the selection limits")
        best = (min if voice["pick"] == "calmest" else max)(eligible, key=lambda candidate: candidate["profile"]["pitch_spread_semitones"])
        wave, sample_rate = sf.read(output_dir / best["wav"], dtype="float32")
        save_voice(voice_name, wave, sample_rate, transcript=config["reference_text"],
                   description=voice["descriptions"][best["description_language"]],
                   metadata={"designed_with": config["model"], "description_language": best["description_language"],
                             "seed": best["seed"], "picked_as": voice["pick"], "profile": best["profile"],
                             "phonetic_error_rate": best["phonetic_error_rate"], "word_error_rate": best["word_error_rate"]})
        chosen[voice_name] = best["wav"]
        print(f"{voice_name} <- {best['wav']}", flush=True)
    return chosen


def render_readme(candidates: list[dict], chosen: dict) -> str:
    rows = [[candidate["voice"], candidate["description_language"], candidate["seed"], candidate["duration_s"],
             candidate["word_error_rate"], candidate["phonetic_error_rate"],
             ", ".join(f"{word['expected']}→{word['heard']}" for word in candidate["flagged_words"]) or "—",
             candidate["profile"]["pitch_median_hz"], candidate["profile"]["pitch_spread_semitones"],
             "**chosen**" if candidate["wav"] in chosen.values() else ""] for candidate in candidates]
    return markdown_table(["voice", "description", "seed", "duration (s)", "WER", "phonetic error", "flagged words",
                           "pitch median (Hz)", "pitch spread (st)", ""], rows)


def main() -> None:
    config = load_config(EXPERIMENT_DIR)
    output_dir = experiment_output_dir(EXPERIMENT_DIR.name)
    candidates = generate_candidates(config, output_dir)
    score_candidates(config, candidates, output_dir)
    chosen = pick_references(config, candidates, output_dir)
    write_results(EXPERIMENT_DIR, config, {"candidates": candidates, "chosen": chosen})
    update_readme_results(EXPERIMENT_DIR, render_readme(candidates, chosen))


if __name__ == "__main__":
    main()
