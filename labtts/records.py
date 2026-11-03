"""Result files and command line shared by the experiments.

Engine comparisons keep one entry per engine in results.json, each with the environment it was produced in, so an
engine can be added or re-run without regenerating the others. README tables are rendered from results.json so the
numbers in the docs never drift from the data.
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

from labtts.paths import REPO_ROOT

TRACKED_PACKAGES = ["qwen-tts", "torch", "transformers", "librosa"]


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
        "packages": {package: importlib.metadata.version(package) for package in TRACKED_PACKAGES},
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", help="comma-separated engine names from config.yaml (default: all of them)")
    parser.add_argument("--rescore", action="store_true",
                        help="judge the audio already in results/ again instead of generating it (e.g. after a judge fix)")
    return parser.parse_args()


def engines_to_run(config: dict, arguments: argparse.Namespace) -> dict:
    """The engines (name -> variants) of config.yaml selected on the command line."""
    if not arguments.engines:
        return config["engines"]
    selected = arguments.engines.split(",")
    unknown = [name for name in selected if name not in config["engines"]]
    if unknown:
        raise SystemExit(f"Not in config.yaml: {', '.join(unknown)}. Configured: {', '.join(config['engines'])}")
    return {name: config["engines"][name] for name in selected}


def load_engine_results(experiment_dir: Path) -> dict:
    """{engine name: {"environment", "voice", "variants", "results"}} from results.json (empty if not run yet)."""
    path = experiment_dir / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))["engines"] if path.exists() else {}


def save_engine_results(experiment_dir: Path, config: dict, engine_name: str, voice: str, results: dict) -> dict:
    """Stores one engine's results, keeping the other engines' entries, and returns all of them."""
    all_engines = load_engine_results(experiment_dir)
    all_engines[engine_name] = {"environment": environment(), "voice": voice,
                                "variants": config["engines"][engine_name], "results": results}
    ordered = {name: all_engines[name] for name in sorted(all_engines, key=list(config["engines"]).index)
               if name in config["engines"]}
    (experiment_dir / "results.json").write_text(
        json.dumps({"config": config, "engines": ordered}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return ordered


def write_results(experiment_dir: Path, config: dict, results: dict) -> Path:
    """results.json for experiments that are not engine comparisons (e.g. 01, which designs the voices)."""
    path = experiment_dir / "results.json"
    path.write_text(json.dumps({"environment": environment(), "config": config, "results": results},
                               indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


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
