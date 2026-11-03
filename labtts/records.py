"""Result files and command line shared by the experiments.

Each experiment keeps one entry per engine in results/<experiment>/results.json (git-ignored, next to the audio),
each with the environment it was produced in, so an engine can be added or re-run without regenerating the others.
README tables are rendered from it, so the numbers in the docs never drift from the data.
"""
import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from labtts.engines import OWN_ENVIRONMENT
from labtts.paths import REPO_ROOT, RESULTS_DIR

TRACKED_PACKAGES = ["qwen-tts", "torch", "transformers", "librosa"]


def _installed_version(package: str) -> str | None:
    """None when the package is absent, as in an engine's separate environment (see README, Adding an engine)."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def environment() -> dict:
    commit = _git("rev-parse", "--short", "HEAD")
    if commit and _git("status", "--porcelain"):
        commit += "-dirty"
    return {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "git_commit": commit or "uncommitted",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "packages": {package: _installed_version(package) for package in TRACKED_PACKAGES},
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", help="comma-separated engine names from config.yaml (default: all of them, except when "
                             "generating those that need their own environment)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rescore", action="store_true",
                      help="judge the audio already in results/ again instead of generating it (e.g. after a judge fix)")
    mode.add_argument("--generate-only", action="store_true",
                      help="generate and store the takes without judging them, for an engine with its own environment; "
                           "judge them afterwards with --rescore in the lab's environment")
    return parser.parse_args()


def engines_to_run(config: dict, arguments: argparse.Namespace) -> dict:
    """The engines (name -> variants) of config.yaml selected on the command line. Without --engines, every engine
    except, when generating, those that need their own environment (their earlier results are kept)."""
    if not arguments.engines:
        return {name: variants for name, variants in config["engines"].items()
                if arguments.rescore or name not in OWN_ENVIRONMENT}
    selected = arguments.engines.split(",")
    unknown = [name for name in selected if name not in config["engines"]]
    if unknown:
        raise SystemExit(f"Not in config.yaml: {', '.join(unknown)}. Configured: {', '.join(config['engines'])}")
    return {name: config["engines"][name] for name in selected}


def results_path(experiment_dir: Path) -> Path:
    """results/<experiment>/results.json: generated next to the audio, so it is git-ignored like the audio."""
    return RESULTS_DIR / experiment_dir.name / "results.json"


def load_engine_results(experiment_dir: Path) -> dict:
    """{engine name: {"environment", "voice", "variants", "results"}} from earlier runs (empty if not run yet)."""
    path = results_path(experiment_dir)
    return json.loads(path.read_text(encoding="utf-8"))["engines"] if path.exists() else {}


def save_engine_results(experiment_dir: Path, config: dict, engine_name: str, voice: str, results: dict) -> dict:
    """Stores one engine's results, keeping the other engines' entries, and returns all of them."""
    all_engines = load_engine_results(experiment_dir)
    all_engines[engine_name] = {"environment": environment(), "voice": voice,
                                "variants": config["engines"][engine_name], "results": results}
    ordered = {name: all_engines[name] for name in sorted(all_engines, key=list(config["engines"]).index)
               if name in config["engines"]}
    path = results_path(experiment_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "engines": ordered}, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return ordered


def voice_used(engine_entry: dict, variant_name: str) -> str:
    """The voice a variant actually spoke with: a variant may switch a cloning engine to its built-in voice."""
    return "built-in" if engine_entry["variants"][variant_name].get("reference") == "none" else engine_entry["voice"]


def format_flagged_words(flagged_words: list[dict], limit: int = 8) -> str:
    shown = ", ".join(f"{word['expected']}→{word['heard']}" for word in flagged_words[:limit])
    remaining = len(flagged_words) - limit
    return (shown + (f" … and {remaining} more" if remaining > 0 else "")) or "—"


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join("" if value is None else str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def update_readme_results(experiment_dir: Path, markdown: str) -> None:
    """Replaces the block between <!-- results:start --> and <!-- results:end --> in the experiment README."""
    readme = experiment_dir / "README.md"
    content = readme.read_text(encoding="utf-8")
    block = f"<!-- results:start -->\n{markdown.strip()}\n<!-- results:end -->"
    updated, replacements = re.subn(r"<!-- results:start -->.*?<!-- results:end -->", lambda _: block, content, flags=re.S)
    if not replacements:
        raise ValueError(f"{readme} has no <!-- results:start --> / <!-- results:end --> markers")
    readme.write_text(updated, encoding="utf-8")
