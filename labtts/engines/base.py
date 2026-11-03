"""The interface every TTS engine implements, so the same experiment can compare different models."""
import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from labtts.voices import Voice


@dataclass
class Synthesis:
    wave: np.ndarray
    sample_rate: int
    generation_seconds: float
    hit_token_limit: bool = False  # the generation was cut by the engine's length cap (a likely runaway)

    @property
    def duration_seconds(self) -> float:
        return len(self.wave) / self.sample_rate

    @property
    def real_time_factor(self) -> float:
        """Generation time divided by audio duration: below 1 is faster than real time."""
        return self.generation_seconds / max(self.duration_seconds, 1e-6)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timed(generate) -> tuple[np.ndarray, int, float]:
    """Runs generate() -> (wave, sample_rate) and returns it with the wall-clock seconds it took on the GPU."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    wave, sample_rate = generate()
    torch.cuda.synchronize()
    wave = wave.detach().float().cpu().numpy() if isinstance(wave, torch.Tensor) else np.asarray(wave, dtype=np.float32)
    return wave.astype(np.float32), int(sample_rate), time.perf_counter() - start


class Engine(ABC):
    """A loaded TTS model ready to speak.

    Engines that can clone use the voice they are given (supports_cloning = True); the others ignore it and speak
    with their built-in voice. Engines that can design a voice from a text description (supports_voice_design =
    True) produce the reference voices under voices/<engine>/ in experiment 01. Settings are engine-specific:
    synthesize() and design_voice() receive the overrides of one variant from the experiment's config.yaml and apply
    them on top of the engine's defaults.
    """
    supports_cloning: bool = False
    supports_voice_design: bool = False

    def __init__(self, name: str, voice: Voice | None):
        self.name = name
        self.voice = voice if self.supports_cloning else None

    @property
    def voice_label(self) -> str:
        return self.voice.name if self.voice else "built-in"

    @abstractmethod
    def synthesize(self, text: str, seed: int, settings: dict | None = None) -> Synthesis:
        """Speaks the whole text in one generation, with the given seed and setting overrides."""

    def design_voice(self, text: str, description: str, seed: int, settings: dict | None = None) -> Synthesis:
        """Speaks the text in a new voice described in natural language."""
        raise NotImplementedError(f"{self.name} cannot design a voice")

    def use_reference(self, reference_audio, transcript: str) -> None:
        """Clones another reference from now on: a file path or a (wave, sample_rate) tuple, with its transcript."""
        raise NotImplementedError(f"{self.name} cannot clone a voice")

    def close(self) -> None:
        """Frees the GPU memory so the next engine or the judges can load."""
        gc.collect()
        torch.cuda.empty_cache()
