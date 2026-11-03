"""Experiment 05: the same everyday sentence with every engine (the samples posted on X).

Each engine and variant says texts/everyday.txt with several seeds. Every take is judged for pronunciation and for
pauses (time spent in silence or breathing), and seed 0 of each variant gets a waveform video ready to share.

Usage: uv run python experiments/05-everyday-sentence/run.py [--engines name,...] [--rescore]
"""
import shutil
import subprocess
from pathlib import Path
from statistics import mean, median

import soundfile as sf

from labtts.engines import load_engine
from labtts.judges.phonetic import PhoneticJudge
from labtts.judges.whisper import WhisperJudge, word_error_rate
from labtts.metrics import analyze_frames, pauses
from labtts.paths import REPO_ROOT, experiment_output_dir
from labtts.records import (engines_to_run, format_flagged_words, load_engine_results, markdown_table,
                            parse_arguments, save_engine_results, update_readme_results, voice_used)
from labtts.texts import load_config, load_text
from labtts.voices import load_voice

EXPERIMENT_DIR = Path(__file__).resolve().parent
VIDEO_TOOL = REPO_ROOT / "tools" / "waveform_video.sh"


def generate(engine_name: str, variants: dict, config: dict, text: str, audio_dir: Path) -> tuple[list[dict], str]:
    engine = load_engine(engine_name, load_voice(config["voice"]))
    takes = []
    for variant_name, overrides in variants.items():
        for seed in config["seeds"]:
            synthesis = engine.synthesize(text, seed, overrides)
            path = audio_dir / f"{variant_name}_seed{seed}.wav"
            sf.write(path, synthesis.wave, synthesis.sample_rate)
            takes.append({"variant": variant_name, "seed": seed, "wav": path.name,
                          "duration_s": round(synthesis.duration_seconds, 2),
                          "real_time_factor": round(synthesis.real_time_factor, 3),
                          "hit_token_limit": synthesis.hit_token_limit})
            print(f"generated {engine_name}/{path.name}: {synthesis.duration_seconds:.1f} s", flush=True)
    voice_label = engine.voice_label
    engine.close()
    return takes, voice_label


def score(config: dict, takes: list[dict], text: str, audio_dir: Path, phonetic_judge, whisper_judge) -> None:
    for take in takes:
        wave, sample_rate = sf.read(audio_dir / take["wav"], dtype="float32")
        verdict = phonetic_judge.evaluate(text, wave, sample_rate)
        transcript = whisper_judge.transcribe(wave, sample_rate)
        take.update({"phonetic_error_rate": round(verdict.error_rate, 4), "flagged_words": verdict.flagged_words,
                     "word_error_rate": round(word_error_rate(text, transcript), 4),
                     "whisper_transcript": transcript,
                     **pauses(analyze_frames(wave, sample_rate), **config["pauses"])})


def render_videos(config: dict, engine_name: str, takes: list[dict], audio_dir: Path) -> None:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found: skipping videos", flush=True)
        return
    for variant_name, (title, subtitle) in config.get("videos", {}).get(engine_name, {}).items():
        take = next((take for take in takes if take["variant"] == variant_name and take["seed"] == 0), None)
        if take:
            subprocess.run([str(VIDEO_TOOL), str(audio_dir / take["wav"]), title, subtitle,
                            str(audio_dir / take["wav"].replace(".wav", ".mp4"))], check=True)


def render_readme(all_engines: dict) -> str:
    groups = [(engine_name, voice_used(entry, variant), variant,
               [take for take in entry["results"]["takes"] if take["variant"] == variant])
              for engine_name, entry in all_engines.items() for variant in entry["variants"]]
    sections = ["### By engine and variant (mean over seeds)", markdown_table(
        ["engine", "variant", "voice", "takes", "duration (s)", "pause time (s)", "pause share", "longest pause (s)",
         "phonetic error", "flagged words", "Whisper WER", "real-time factor"],
        [[engine_name, variant, voice, len(takes), round(mean(take["duration_s"] for take in takes), 1),
          round(mean(take["pause_seconds"] for take in takes), 1), f"{mean(take['pause_share'] for take in takes):.0%}",
          round(mean(take["longest_pause_seconds"] for take in takes), 2),
          round(mean(take["phonetic_error_rate"] for take in takes), 4),
          ", ".join(f"{word['expected']}→{word['heard']}" for take in takes for word in take["flagged_words"]) or "—",
          round(mean(take["word_error_rate"] for take in takes), 4),
          round(median(take["real_time_factor"] for take in takes), 2)] for engine_name, voice, variant, takes in groups])]
    sections += ["### Every take", markdown_table(
        ["engine", "variant", "seed", "duration (s)", "pauses", "pause time (s)", "phonetic error", "Whisper heard"],
        [[engine_name, take["variant"], take["seed"], take["duration_s"], take["pauses"], take["pause_seconds"],
          take["phonetic_error_rate"], take["whisper_transcript"]]
         for engine_name, entry in all_engines.items() for take in entry["results"]["takes"]])]
    return "\n\n".join(sections)


def main() -> None:
    arguments = parse_arguments()
    config = load_config(EXPERIMENT_DIR)
    text = load_text(config["text"])
    previous = load_engine_results(EXPERIMENT_DIR)
    selected = {}
    for engine_name, variants in engines_to_run(config, arguments).items():
        audio_dir = experiment_output_dir(f"{EXPERIMENT_DIR.name}/{engine_name}")
        selected[engine_name] = ((previous[engine_name]["results"]["takes"], previous[engine_name]["voice"]) if arguments.rescore
                                 else generate(engine_name, variants, config, text, audio_dir)) + (audio_dir,)
    if arguments.generate_only:  # judged later, with --rescore in the lab's environment
        for engine_name, (takes, voice_label, _) in selected.items():
            save_engine_results(EXPERIMENT_DIR, config, engine_name, voice_label, {"takes": takes})
        return
    # the judges load once, after every engine has released the GPU
    phonetic_judge, whisper_judge = PhoneticJudge(), WhisperJudge()
    all_engines = previous
    for engine_name, (takes, voice_label, audio_dir) in selected.items():
        score(config, takes, text, audio_dir, phonetic_judge, whisper_judge)
        all_engines = save_engine_results(EXPERIMENT_DIR, config, engine_name, voice_label, {"takes": takes})
        render_videos(config, engine_name, takes, audio_dir)
    update_readme_results(EXPERIMENT_DIR, render_readme(all_engines))


if __name__ == "__main__":
    main()
