"""Experiment 03: does the voice hold up over two minutes, and is it better to generate in one go or sentence by
sentence?

For each engine, variant and seed the same ~2 min text is generated in one call and as one call per sentence (joined
with a short silence). Both are judged for pronunciation, and pitch, loudness and speaking rate are tracked every
20 s to see whether the voice drifts.

Usage: uv run python experiments/03-long-form/run.py [--engines name,...] [--rescore]
"""
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf

from labtts.engines import load_engine
from labtts.judges.phonetic import PhoneticJudge
from labtts.judges.whisper import WhisperJudge, word_error_rate
from labtts.metrics import analyze_frames, windows_over_time
from labtts.paths import experiment_output_dir
from labtts.records import (engines_to_run, format_flagged_words, load_engine_results, markdown_table,
                            parse_arguments, save_engine_results, update_readme_results, voice_used)
from labtts.texts import load_config, load_text, split_sentences
from labtts.voices import load_voice

EXPERIMENT_DIR = Path(__file__).resolve().parent


def generate(engine_name: str, variants: dict, config: dict, text: str, audio_dir: Path) -> tuple[list[dict], str]:
    engine = load_engine(engine_name, load_voice(config["voice"]))
    runs = []
    for variant_name, overrides in variants.items():
        for seed in config["seeds"]:
            for mode in config["modes"]:
                if mode == "one_shot":
                    synthesis = engine.synthesize(text, seed, overrides)
                    wave, sample_rate = synthesis.wave, synthesis.sample_rate
                    generation_seconds, hit_token_limit = synthesis.generation_seconds, synthesis.hit_token_limit
                else:
                    pieces, generation_seconds, hit_token_limit = [], 0.0, False
                    for sentence_index, sentence in enumerate(split_sentences(text)):
                        synthesis = engine.synthesize(sentence, seed * 1000 + sentence_index, overrides)
                        sample_rate = synthesis.sample_rate
                        pieces += [synthesis.wave, np.zeros(int(config["sentence_gap_seconds"] * sample_rate), dtype=np.float32)]
                        generation_seconds += synthesis.generation_seconds
                        hit_token_limit |= synthesis.hit_token_limit
                    wave = np.concatenate(pieces[:-1])
                path = audio_dir / f"{variant_name}_{mode}_seed{seed}.wav"
                sf.write(path, wave, sample_rate)
                duration = len(wave) / sample_rate
                runs.append({"variant": variant_name, "mode": mode, "seed": seed, "wav": path.name,
                             "duration_s": round(duration, 1), "real_time_factor": round(generation_seconds / duration, 3),
                             "hit_token_limit": hit_token_limit})
                print(f"generated {engine_name}/{path.name}: {duration:.1f} s", flush=True)
    voice_label = engine.voice_label
    engine.close()
    return runs, voice_label


def score(config: dict, runs: list[dict], text: str, audio_dir: Path, phonetic_judge, whisper_judge) -> None:
    for run in runs:
        wave, sample_rate = sf.read(audio_dir / run["wav"], dtype="float32")
        verdict = phonetic_judge.evaluate(text, wave, sample_rate)
        transcript = whisper_judge.transcribe(wave, sample_rate)
        # speaking rate uses the words the CTC recognizer heard and when each starts (Whisper word timing is too costly)
        windows = windows_over_time(analyze_frames(wave, sample_rate), config["window_seconds"],
                                    verdict.heard_words, verdict.word_start_seconds)
        pitches = [window["pitch_median_hz"] for window in windows if window["pitch_median_hz"]]
        # an engine that stops early may leave fewer than two windows: there is no drift to measure then
        run.update({"phonetic_error_rate": round(verdict.error_rate, 4), "flagged_words": verdict.flagged_words,
                    "word_error_rate": round(word_error_rate(text, transcript), 4), "windows": windows,
                    "loudness_drift_db": round(windows[-1]["loudness_dbfs"] - windows[0]["loudness_dbfs"], 1)
                    if len(windows) > 1 else None,
                    "pitch_range_hz": [min(pitches), max(pitches)] if pitches else None})


def render_readme(config: dict, all_engines: dict) -> str:
    every_run = [(engine_name, run) for engine_name, entry in all_engines.items() for run in entry["results"]["runs"]]
    sections = ["### Pronunciation and drift", markdown_table(
        ["engine", "variant", "mode", "seed", "duration (s)", "phonetic error", "flagged words", "Whisper WER",
         "loudness last − first window (dB)", "window pitch range (Hz)"],
        [[engine_name, run["variant"], run["mode"], run["seed"], run["duration_s"], run["phonetic_error_rate"],
          format_flagged_words(run["flagged_words"]),
          run["word_error_rate"], "—" if run["loudness_drift_db"] is None else run["loudness_drift_db"],
          f"{run['pitch_range_hz'][0]}–{run['pitch_range_hz'][1]}" if run["pitch_range_hz"] else "—"]
         for engine_name, run in every_run])]
    drifts = {}
    for engine_name, run in every_run:
        if run["loudness_drift_db"] is not None:
            drifts.setdefault((engine_name, run["variant"], run["mode"]), []).append(run["loudness_drift_db"])
    groups = dict.fromkeys((engine_name, run["variant"], run["mode"]) for engine_name, run in every_run)
    sections += ["### Mean over seeds", markdown_table(
        ["engine", "variant", "mode", "phonetic error", "loudness drift (dB)"],
        [[engine_name, variant, mode,
          round(mean(run["phonetic_error_rate"] for name, run in every_run if (name, run["variant"], run["mode"]) == (engine_name, variant, mode)), 4),
          round(mean(drifts[(engine_name, variant, mode)]), 1) if (engine_name, variant, mode) in drifts else "—"]
         for engine_name, variant, mode in groups])]
    for engine_name, run in every_run:
        if run["seed"] != config["seeds"][0]:
            continue
        sections += [f"### Every {config['window_seconds']} s · {engine_name}, {run['variant']}, {run['mode']}, seed {run['seed']}",
                     markdown_table(["window", "pitch median (Hz)", "loudness (dBFS)", "syllables / s"],
                                    [[f"{window['start_s']:.0f}–{window['end_s']:.0f} s", window["pitch_median_hz"],
                                      window["loudness_dbfs"], window["syllables_per_second"]] for window in run["windows"]])]
    return "\n\n".join(sections)


def main() -> None:
    arguments = parse_arguments()
    config = load_config(EXPERIMENT_DIR)
    text = load_text(config["text"])
    previous = load_engine_results(EXPERIMENT_DIR)
    selected = {}
    for engine_name, variants in engines_to_run(config, arguments).items():
        audio_dir = experiment_output_dir(f"{EXPERIMENT_DIR.name}/{engine_name}")
        selected[engine_name] = ((previous[engine_name]["results"]["runs"], previous[engine_name]["voice"]) if arguments.rescore
                                 else generate(engine_name, variants, config, text, audio_dir)) + (audio_dir,)
    # the judges load once, after every engine has released the GPU
    phonetic_judge, whisper_judge = PhoneticJudge(), WhisperJudge()
    all_engines = previous
    for engine_name, (runs, voice_label, audio_dir) in selected.items():
        score(config, runs, text, audio_dir, phonetic_judge, whisper_judge)
        all_engines = save_engine_results(EXPERIMENT_DIR, config, engine_name, voice_label, {"runs": runs})
    update_readme_results(EXPERIMENT_DIR, render_readme(config, all_engines))


if __name__ == "__main__":
    main()
