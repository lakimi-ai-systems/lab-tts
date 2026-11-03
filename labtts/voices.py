"""Reference voices stored under voices/<name>/: reference.wav, transcript.txt, description.txt, metadata.json.

All voices in this repo are synthetic (designed from a text description), so no real person is cloned.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from labtts.paths import VOICES_DIR


@dataclass
class Voice:
    name: str
    reference_audio: Path
    transcript: str
    description: str


def load_voice(name: str) -> Voice:
    voice_dir = VOICES_DIR / name
    if not (voice_dir / "reference.wav").exists():
        raise FileNotFoundError(f"Voice '{name}' not found in {voice_dir}. Run experiments/01-voice-design first.")
    description_file = voice_dir / "description.txt"
    return Voice(name=name, reference_audio=voice_dir / "reference.wav",
                 transcript=(voice_dir / "transcript.txt").read_text(encoding="utf-8").strip(),
                 description=description_file.read_text(encoding="utf-8").strip() if description_file.exists() else "")


def save_voice(name: str, wave: np.ndarray, sample_rate: int, transcript: str, description: str, metadata: dict) -> Voice:
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)
    sf.write(voice_dir / "reference.wav", wave, sample_rate)
    (voice_dir / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    (voice_dir / "description.txt").write_text(description + "\n", encoding="utf-8")
    (voice_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return load_voice(name)
