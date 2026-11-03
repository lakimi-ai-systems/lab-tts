"""Qwen3-TTS (12 Hz): Base models for cloning (Qwen3TTSEngine) and the VoiceDesign model for designing reference
voices (Qwen3VoiceDesignEngine).

Requires the qwen-tts package (a core dependency of this repo).
"""
from dataclasses import asdict, dataclass

import torch

from labtts.engines.base import Engine, Synthesis, set_seed, timed
from labtts.voices import Voice

BASE_MODELS = {"0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
# The 12 Hz tokenizer emits 12.5 codec frames per second of audio
CODEC_FRAMES_PER_SECOND = 12.5


@dataclass
class QwenSampling:
    """Generation settings. Defaults match the models' generation_config.json, except max_new_tokens: the shipped
    value (8192, ~11 min of audio) lets a rare runaway generation babble for minutes, so it is capped explicitly.

    The talker samples one semantic token per frame (content and prosody); the subtalker samples the remaining
    acoustic codebooks of that frame (fine timbre detail). Each has its own temperature and top-k.
    """
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    max_new_tokens: int = 2048

    @classmethod
    def from_overrides(cls, overrides: dict | None) -> "QwenSampling":
        return cls(**(overrides or {}))  # an unknown key raises, so a typo in config.yaml does not pass silently

    def as_kwargs(self) -> dict:
        return {"do_sample": True, "subtalker_dosample": True, **asdict(self)}


def load_model(model_id: str):
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(model_id, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")


def _synthesis(generate, sampling: QwenSampling) -> Synthesis:
    def first_wave():
        waves, sample_rate = generate()
        return waves[0], sample_rate
    wave, sample_rate, seconds = timed(first_wave)
    frames = len(wave) / sample_rate * CODEC_FRAMES_PER_SECOND
    return Synthesis(wave=wave, sample_rate=sample_rate, generation_seconds=seconds,
                     hit_token_limit=frames >= sampling.max_new_tokens - 1)


class Qwen3VoiceDesignEngine(Engine):
    """The 1.7B VoiceDesign model: speech in a voice described in natural language, with no reference audio."""
    supports_voice_design = True

    def __init__(self, name: str, voice: Voice | None):
        super().__init__(name, voice)
        self.model = load_model(VOICE_DESIGN_MODEL)

    def design_voice(self, text: str, description: str, seed: int, settings: dict | None = None) -> Synthesis:
        sampling = QwenSampling.from_overrides(settings)
        set_seed(seed)
        return _synthesis(lambda: self.model.generate_voice_design(text=text, instruct=description, language="Spanish",
                                                                   **sampling.as_kwargs()), sampling)

    def synthesize(self, text: str, seed: int, settings: dict | None = None) -> Synthesis:
        """Without a description the model picks a voice itself (an empty instruction is allowed)."""
        return self.design_voice(text, "", seed, settings)

    def close(self) -> None:
        del self.model
        super().close()


class Qwen3TTSEngine(Engine):
    """A Base model plus a voice-clone prompt built once and reused for every generation.

    The prompt uses in-context mode (reference audio codes + exact transcript), not the speaker embedding alone.
    """
    supports_cloning = True

    def __init__(self, name: str, voice: Voice | None, model_size: str):
        super().__init__(name, voice)
        if voice is None:
            raise ValueError(f"{name} only speaks by cloning: give it a reference voice")
        self.model = load_model(BASE_MODELS[model_size])
        self.use_reference(str(voice.reference_audio), voice.transcript)

    def use_reference(self, reference_audio, transcript: str) -> None:
        self.prompt = self.model.create_voice_clone_prompt(ref_audio=reference_audio, ref_text=transcript)

    def synthesize(self, text: str, seed: int, settings: dict | None = None) -> Synthesis:
        sampling = QwenSampling.from_overrides(settings)
        set_seed(seed)
        return _synthesis(lambda: self.model.generate_voice_clone(
            text=text, language="Spanish", voice_clone_prompt=self.prompt, **sampling.as_kwargs()), sampling)

    def close(self) -> None:
        del self.model
        super().close()
