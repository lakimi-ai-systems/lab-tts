"""Experiment 01: design voices from a text description and pick the reference take automatically.

1. Every voice-design engine says the same ~10 s sentence for every description, variant and seed.
2. Each take is scored: Whisper WER, phonetic judge, pitch profile.
3. Among the takes that are fully intelligible, the calmest (lowest pitch spread) becomes
   voices/<engine>/warm_male and the liveliest of the energetic description becomes voices/<engine>/warm_male_energetic.

Usage: uv run python experiments/01-voice-design/run.py [--engines name,...] [--rescore]   (overwrites voices/<engine>/)
"""
from pathlib import Path

import soundfile as sf

from labtts.engines import load_engine
from labtts.judges.phonetic import PhoneticJudge
from labtts.judges.whisper import WhisperJudge, word_error_rate
from labtts.metrics import analyze_frames, voice_profile
from labtts.paths import experiment_output_dir
from labtts.records import (engines_to_run, format_flagged_words, load_engine_results, markdown_table,
                            parse_arguments, save_engine_results, update_readme_results)
from labtts.texts import load_config
from labtts.voices import save_voice

EXPERIMENT_DIR = Path(__file__).resolve().parent


def generate_candidates(engine_name: str, variants: dict, config: dict, audio_dir: Path) -> list[dict] | None:
    engine = load_engine(engine_name)
    if not engine.supports_voice_design:
        print(f"skipping {engine_name}: it cannot design a voice", flush=True)
        engine.close()
        return None
    candidates = []
    for variant_name, overrides in variants.items():
        for voice_name, voice in config["voices"].items():
            for description_language, description in voice["descriptions"].items():
                for seed in config["seeds"]:
                    synthesis = engine.design_voice(config["reference_text"], description, seed, overrides)
                    path = audio_dir / f"{variant_name}_{voice_name}_{description_language}_seed{seed}.wav"
                    sf.write(path, synthesis.wave, synthesis.sample_rate)
                    candidates.append({"variant": variant_name, "voice": voice_name,
                                       "description_language": description_language, "seed": seed, "wav": path.name,
                                       "duration_s": round(synthesis.duration_seconds, 2),
                                       "real_time_factor": round(synthesis.real_time_factor, 3)})
                    print(f"generated {engine_name}/{path.name}: {synthesis.duration_seconds:.1f} s", flush=True)
    engine.close()
    return candidates


def score_candidates(config: dict, candidates: list[dict], audio_dir: Path, phonetic_judge, whisper_judge) -> None:
    for candidate in candidates:
        wave, sample_rate = sf.read(audio_dir / candidate["wav"], dtype="float32")
        verdict = phonetic_judge.evaluate(config["reference_text"], wave, sample_rate)
        transcript = whisper_judge.transcribe(wave, sample_rate)
        profile = voice_profile(analyze_frames(wave, sample_rate))
        candidate.update({"phonetic_error_rate": round(verdict.error_rate, 4), "flagged_words": verdict.flagged_words,
                          "word_error_rate": round(word_error_rate(config["reference_text"], transcript), 4),
                          "whisper_transcript": transcript, "profile": profile.as_dict()})


def pick_references(engine_name: str, config: dict, candidates: list[dict], audio_dir: Path) -> dict:
    limits = config["selection"]
    chosen = {}
    for voice_name, voice in config["voices"].items():
        eligible = [candidate for candidate in candidates if candidate["voice"] == voice_name
                    and candidate["word_error_rate"] <= limits["max_word_error_rate"]
                    and len(candidate["flagged_words"]) <= limits["max_flagged_words"]]
        if not eligible:
            print(f"no intelligible take for {engine_name}/{voice_name}: add seeds or relax the selection limits", flush=True)
            continue
        best = (min if voice["pick"] == "calmest" else max)(eligible, key=lambda candidate: candidate["profile"]["pitch_spread_semitones"])
        wave, sample_rate = sf.read(audio_dir / best["wav"], dtype="float32")
        save_voice(f"{engine_name}/{voice_name}", wave, sample_rate, transcript=config["reference_text"],
                   description=voice["descriptions"][best["description_language"]],
                   metadata={"designed_with": engine_name, "variant": best["variant"],
                             "description_language": best["description_language"], "seed": best["seed"],
                             "picked_as": voice["pick"], "profile": best["profile"],
                             "phonetic_error_rate": best["phonetic_error_rate"], "word_error_rate": best["word_error_rate"]})
        chosen[voice_name] = best["wav"]
        print(f"voices/{engine_name}/{voice_name} <- {best['wav']}", flush=True)
    return chosen


def render_readme(all_engines: dict) -> str:
    rows = [[engine_name, candidate["voice"], candidate["description_language"], candidate["seed"],
             candidate["duration_s"], candidate["word_error_rate"], candidate["phonetic_error_rate"],
             format_flagged_words(candidate["flagged_words"]), candidate["profile"]["pitch_median_hz"],
             candidate["profile"]["pitch_spread_semitones"],
             "**chosen**" if candidate["wav"] in entry["results"]["chosen"].values() else ""]
            for engine_name, entry in all_engines.items() for candidate in entry["results"]["candidates"]]
    return markdown_table(["designer", "voice", "description", "seed", "duration (s)", "WER", "phonetic error",
                           "flagged words", "pitch median (Hz)", "pitch spread (st)", ""], rows)


def main() -> None:
    arguments = parse_arguments()
    config = load_config(EXPERIMENT_DIR)
    previous = load_engine_results(EXPERIMENT_DIR)
    selected = {}
    for engine_name, variants in engines_to_run(config, arguments).items():
        audio_dir = experiment_output_dir(f"{EXPERIMENT_DIR.name}/{engine_name}")
        candidates = (previous[engine_name]["results"]["candidates"] if arguments.rescore
                      else generate_candidates(engine_name, variants, config, audio_dir))
        if candidates is not None:
            selected[engine_name] = (candidates, audio_dir)
    # the judges load once, after every engine has released the GPU
    phonetic_judge, whisper_judge = PhoneticJudge(), WhisperJudge()
    all_engines = previous
    for engine_name, (candidates, audio_dir) in selected.items():
        score_candidates(config, candidates, audio_dir, phonetic_judge, whisper_judge)
        chosen = pick_references(engine_name, config, candidates, audio_dir)
        all_engines = save_engine_results(EXPERIMENT_DIR, config, engine_name, "designed",
                                          {"candidates": candidates, "chosen": chosen})
    update_readme_results(EXPERIMENT_DIR, render_readme(all_engines))


if __name__ == "__main__":
    main()
