import json
import re

import yaml

from labtts.paths import TEXTS_DIR


def load_text(name: str) -> str:
    """A plain-text file from texts/, with line breaks joined into a single paragraph."""
    return " ".join((TEXTS_DIR / name).read_text(encoding="utf-8").split())


def load_items(name: str) -> list[dict]:
    """A JSON Lines file from texts/: one {"id": ..., "text": ...} per line."""
    return [json.loads(line) for line in (TEXTS_DIR / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def split_sentences(text: str) -> list[str]:
    """Splits after . : ? ! — the way a voice gateway flushes text to the TTS one sentence at a time."""
    return [sentence.strip() for sentence in re.split(r"(?<=[.:?!])\s+", text) if sentence.strip()]


def load_config(experiment_dir) -> dict:
    return yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))
