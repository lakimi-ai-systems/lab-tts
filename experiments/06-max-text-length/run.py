"""Experiment 06: how much text can each engine read in a single generation?

Each take reads the first N sentences of texts/long_form_ai.txt in one call. Whisper's WER against the requested
text tells whether the engine read all of it, and the take records whether the generation ended on its own or ran
into the engine's length cap.

Usage: uv run python experiments/06-max-text-length/run.py [--engines name,...] [--rescore]
"""
from pathlib import Path
from statistics import mean

import soundfile as sf

from labtts.engines import load_engine
from labtts.judges.whisper import WhisperJudge, word_error_rate
from labtts.paths import experiment_output_dir
from labtts.records import (engines_to_run, load_engine_results, markdown_table, parse_arguments, save_engine_results,
                            update_readme_results, voice_used)
from labtts.texts import load_config, load_text, split_sentences
from labtts.voices import load_voice

EXPERIMENT_DIR = Path(__file__).resolve().parent


def requested_text(config: dict, sentence_count: int) -> str:
    return " ".join(split_sentences(load_text(config["text"]))[:sentence_count])


def generate(engine_name: str, variants: dict, config: dict, audio_dir: Path) -> tuple[list[dict], str]:
    engine = load_engine(engine_name, load_voice(config["voice"]))
    takes = []
    for variant_name, overrides in variants.items():
        for sentence_count in config["sentence_counts"]:
            text = requested_text(config, sentence_count)
            for seed in config["seeds"]:
                synthesis = engine.synthesize(text, seed, overrides)
                path = audio_dir / f"{variant_name}_{sentence_count:02d}_sentences_seed{seed}.wav"
                sf.write(path, synthesis.wave, synthesis.sample_rate)
                takes.append({"variant": variant_name, "sentences": sentence_count, "words": len(text.split()),
                              "seed": seed, "wav": path.name, "duration_s": round(synthesis.duration_seconds, 1),
                              "hit_token_limit": synthesis.hit_token_limit})
                print(f"generated {engine_name}/{path.name}: {synthesis.duration_seconds:.1f} s", flush=True)
    voice_label = engine.voice_label
    engine.close()
    return takes, voice_label


def score(config: dict, takes: list[dict], audio_dir: Path, whisper_judge) -> None:
    for take in takes:
        wave, sample_rate = sf.read(audio_dir / take["wav"], dtype="float32")
        transcript = whisper_judge.transcribe(wave, sample_rate)
        error_rate = word_error_rate(requested_text(config, take["sentences"]), transcript)
        take.update({"word_error_rate": round(error_rate, 4), "whisper_transcript": transcript,
                     "read_in_full": error_rate <= config["max_word_error_rate_read_in_full"]})


def render_readme(all_engines: dict) -> str:
    rows = []
    for engine_name, entry in all_engines.items():
        takes = entry["results"]["takes"]
        for variant_name in entry["variants"]:
            for sentence_count in dict.fromkeys(take["sentences"] for take in takes if take["variant"] == variant_name):
                group = [take for take in takes if take["variant"] == variant_name and take["sentences"] == sentence_count]
                rows.append([engine_name, variant_name, voice_used(entry, variant_name), sentence_count, group[0]["words"],
                             f"{sum(take['read_in_full'] for take in group)}/{len(group)}",
                             round(mean(take["duration_s"] for take in group), 1),
                             round(mean(take["word_error_rate"] for take in group), 3),
                             sum(take["hit_token_limit"] for take in group)])
    return markdown_table(["engine", "variant", "voice", "sentences", "words", "read in full", "duration (s)",
                           "Whisper WER", "hit length cap"], rows)


def main() -> None:
    arguments = parse_arguments()
    config = load_config(EXPERIMENT_DIR)
    previous = load_engine_results(EXPERIMENT_DIR)
    selected = {}
    for engine_name, variants in engines_to_run(config, arguments).items():
        audio_dir = experiment_output_dir(f"{EXPERIMENT_DIR.name}/{engine_name}")
        selected[engine_name] = ((previous[engine_name]["results"]["takes"], previous[engine_name]["voice"]) if arguments.rescore
                                 else generate(engine_name, variants, config, audio_dir)) + (audio_dir,)
    whisper_judge = WhisperJudge()  # loads once, after every engine has released the GPU
    all_engines = previous
    for engine_name, (takes, voice_label, audio_dir) in selected.items():
        score(config, takes, audio_dir, whisper_judge)
        all_engines = save_engine_results(EXPERIMENT_DIR, config, engine_name, voice_label, {"takes": takes})
    update_readme_results(EXPERIMENT_DIR, render_readme(all_engines))


if __name__ == "__main__":
    main()
