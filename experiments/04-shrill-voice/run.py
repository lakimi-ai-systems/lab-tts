"""Experiment 04: why does a cloned voice sometimes turn shrill?

Four suspects, each tested against the calm target voice:
  reference          a calm reference versus a lively, high-pitched one
  sampling           the engine's variants in config.yaml (for Qwen3-TTS: talker and subtalker temperature)
  re-cloning         using each generated output as the reference for the next one

Every generation is split into one-second windows; a window counts as shrill when its pitch rises more than a few
semitones above the target voice or its brightness (spectral centroid) clearly exceeds it. See labtts.metrics.

Usage: uv run python experiments/04-shrill-voice/run.py [--engines name,...] [--rescore]
"""
from pathlib import Path
from statistics import mean

import soundfile as sf

from labtts.engines import load_engine
from labtts.judges.phonetic import PhoneticJudge
from labtts.metrics import ShrillnessThresholds, analyze_frames, semitones_between, shrill_windows, voice_profile
from labtts.paths import experiment_output_dir
from labtts.records import (engines_to_run, load_engine_results, markdown_table, parse_arguments, save_engine_results,
                            update_readme_results)
from labtts.texts import load_config, load_items
from labtts.voices import load_voice

EXPERIMENT_DIR = Path(__file__).resolve().parent


def generate(engine_name: str, variants: dict, config: dict, paragraphs: list[dict], audio_dir: Path) -> list[dict] | None:
    target = load_voice(config["target_voice"])
    engine = load_engine(engine_name, target)
    if not engine.supports_cloning:
        print(f"skipping {engine_name}: this experiment is about cloning and it cannot clone", flush=True)
        engine.close()
        return None
    generations = []
    for reference_name in config["references"]:
        reference = load_voice(reference_name)
        engine.use_reference(str(reference.reference_audio), reference.transcript)
        for variant_name, overrides in variants.items():
            for seed, paragraph in enumerate(paragraphs):
                synthesis = engine.synthesize(paragraph["text"], seed, overrides)
                path = audio_dir / f"{reference_name}_{variant_name}_{paragraph['id']}.wav"
                sf.write(path, synthesis.wave, synthesis.sample_rate)
                generations.append({"group": "grid", "reference": reference_name, "variant": variant_name,
                                    "text_id": paragraph["id"], "text": paragraph["text"], "wav": path.name,
                                    "duration_s": round(synthesis.duration_seconds, 2),
                                    "hit_token_limit": synthesis.hit_token_limit})
                print(f"generated {engine_name}/{path.name}: {synthesis.duration_seconds:.1f} s", flush=True)

    engine.use_reference(str(target.reference_audio), target.transcript)
    for step in range(config["reclone_chain"]["steps"]):
        paragraph = paragraphs[step % len(paragraphs)]
        synthesis = engine.synthesize(paragraph["text"], step)
        path = audio_dir / f"reclone_step{step + 1}_{paragraph['id']}.wav"
        sf.write(path, synthesis.wave, synthesis.sample_rate)
        generations.append({"group": "reclone_chain", "step": step + 1, "text_id": paragraph["id"],
                            "text": paragraph["text"], "wav": path.name,
                            "duration_s": round(synthesis.duration_seconds, 2), "hit_token_limit": synthesis.hit_token_limit})
        print(f"generated {engine_name}/{path.name}: {synthesis.duration_seconds:.1f} s", flush=True)
        # the next generation clones this output instead of the original reference
        engine.use_reference((synthesis.wave, synthesis.sample_rate), paragraph["text"])
    engine.close()
    return generations


def reference_profiles(config: dict) -> dict:
    names = dict.fromkeys([config["target_voice"], *config["references"]])
    return {name: voice_profile(analyze_frames(*sf.read(load_voice(name).reference_audio, dtype="float32"))) for name in names}


def score(config: dict, generations: list[dict], audio_dir: Path, profiles: dict, phonetic_judge) -> None:
    thresholds = ShrillnessThresholds(**config["shrillness"])
    target_profile = profiles[config["target_voice"]]
    for generation in generations:
        wave, sample_rate = sf.read(audio_dir / generation["wav"], dtype="float32")
        analysis = analyze_frames(wave, sample_rate)
        profile = voice_profile(analysis)
        # the re-cloning chain starts from the target voice; grid generations are compared with the reference they cloned
        own_reference = profiles[generation.get("reference", config["target_voice"])]
        generation.update({
            "phonetic_error_rate": round(phonetic_judge.evaluate(generation["text"], wave, sample_rate).error_rate, 4),
            "profile": profile.as_dict(),
            "pitch_median_shift_st": round(semitones_between(profile.pitch_median_hz, target_profile.pitch_median_hz), 2),
            "pitch_p95_shift_st": round(semitones_between(profile.pitch_p95_hz, target_profile.pitch_p95_hz), 2),
            "centroid_ratio": round(profile.centroid_median_hz / target_profile.centroid_median_hz, 3),
            "pitch_median_vs_own_reference_st": round(semitones_between(profile.pitch_median_hz, own_reference.pitch_median_hz), 2),
            "centroid_vs_own_reference": round(profile.centroid_median_hz / own_reference.centroid_median_hz, 3),
            **shrill_windows(analysis, target_profile, thresholds)})


def summarize(config: dict, variants: dict, generations: list[dict]) -> list[dict]:
    rows = []
    for reference_name in config["references"]:
        for variant_name in variants:
            group = [generation for generation in generations if generation["group"] == "grid"
                     and generation["reference"] == reference_name and generation["variant"] == variant_name]

            def average(key: str, digits: int) -> float:
                return round(mean(generation[key] for generation in group), digits)
            rows.append({"reference": reference_name, "variant": variant_name, "generations": len(group),
                         "shrill_share_mean": average("shrill_share", 4),
                         "generations_with_shrill_windows": sum(generation["shrill_windows"] > 0 for generation in group),
                         "pitch_median_shift_st": average("pitch_median_shift_st", 2),
                         "pitch_p95_shift_st": average("pitch_p95_shift_st", 2),
                         "centroid_ratio": average("centroid_ratio", 3),
                         "pitch_median_vs_own_reference_st": average("pitch_median_vs_own_reference_st", 2),
                         "centroid_vs_own_reference": average("centroid_vs_own_reference", 3),
                         "phonetic_error_rate": average("phonetic_error_rate", 4),
                         "hit_token_limit": sum(generation["hit_token_limit"] for generation in group)})
    return rows


def render_readme(config: dict, profiles: dict, all_engines: dict) -> str:
    sections = ["### Reference voices", markdown_table(
        ["voice", "pitch median (Hz)", "pitch p95 (Hz)", "pitch spread (st)", "brightness median (Hz)"],
        [[name, *(profile.as_dict()[key] for key in ("pitch_median_hz", "pitch_p95_hz", "pitch_spread_semitones",
                                                    "centroid_median_hz"))] for name, profile in profiles.items()])]
    sections += [f"### Reference × variant ({len(load_items(config['texts']))} paragraphs each)", markdown_table(
        ["engine", "reference", "variant", "shrill windows (mean share)", "generations with shrill windows",
         "pitch median vs target (st)", "pitch p95 vs target (st)", "brightness vs target",
         "pitch median vs own reference (st)", "brightness vs own reference", "phonetic error"],
        [[engine_name, row["reference"], row["variant"], f"{row['shrill_share_mean']:.1%}",
          f"{row['generations_with_shrill_windows']}/{row['generations']}", row["pitch_median_shift_st"],
          row["pitch_p95_shift_st"], row["centroid_ratio"], row["pitch_median_vs_own_reference_st"],
          row["centroid_vs_own_reference"], row["phonetic_error_rate"]]
         for engine_name, entry in all_engines.items() for row in entry["results"]["summary"]])]
    sections += ["### Re-cloning chain (each step clones the previous output)", markdown_table(
        ["engine", "step", "shrill windows", "pitch median vs target (st)", "pitch p95 vs target (st)",
         "brightness vs target", "phonetic error"],
        [[engine_name, generation["step"], f"{generation['shrill_share']:.1%}", generation["pitch_median_shift_st"],
          generation["pitch_p95_shift_st"], generation["centroid_ratio"], generation["phonetic_error_rate"]]
         for engine_name, entry in all_engines.items() for generation in entry["results"]["generations"]
         if generation["group"] == "reclone_chain"])]
    return "\n\n".join(sections)


def main() -> None:
    arguments = parse_arguments()
    config = load_config(EXPERIMENT_DIR)
    paragraphs = load_items(config["texts"])
    previous = load_engine_results(EXPERIMENT_DIR)
    selected = {}
    for engine_name, variants in engines_to_run(config, arguments).items():
        audio_dir = experiment_output_dir(f"{EXPERIMENT_DIR.name}/{engine_name}")
        generations = (previous[engine_name]["results"]["generations"] if arguments.rescore
                       else generate(engine_name, variants, config, paragraphs, audio_dir))
        if generations is not None:
            selected[engine_name] = (variants, generations, audio_dir)
    profiles = reference_profiles(config)
    phonetic_judge = PhoneticJudge()  # loads once, after every engine has released the GPU
    all_engines = previous
    for engine_name, (variants, generations, audio_dir) in selected.items():
        score(config, generations, audio_dir, profiles, phonetic_judge)
        all_engines = save_engine_results(EXPERIMENT_DIR, config, engine_name, config["target_voice"],
                                          {"summary": summarize(config, variants, generations), "generations": generations})
    update_readme_results(EXPERIMENT_DIR, render_readme(config, profiles, all_engines))


if __name__ == "__main__":
    main()
