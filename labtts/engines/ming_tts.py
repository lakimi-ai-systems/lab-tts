"""Ming-3.0-tts-0.5B (inclusionAI): a Qwen2.5-0.5B backbone with a flow-matching head over continuous 44.1 kHz VAE
latents, zero-shot cloning in seven languages, Spanish among them.

It needs its own environment (torch 2.6, transformers 4.46.3) and its inference code, which is not on PyPI: clone
https://github.com/inclusionAI/Ming-3.0-tts at the commit below, point MING_TTS_CODE at it, generate there and judge
here with --rescore (see "Adding an engine" in the root README).

Context: prompt, reference and generated latents share 512 positions, one latent being 0.08 s of audio, so with a
~10 s reference a single generation stops at about 20 s of speech. The model's own API therefore splits the text at
punctuation into fragments of about `max_length` units and speaks each with the same reference: that is the default
here (`split_text: true`). `split_text: false` sends the whole text to one generation, as the published benchmarks
do; Synthesis.hit_token_limit records a generation that ran out of positions (or a text that alone fills them, which
yields an empty take).
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

from labtts.engines.base import Engine, Synthesis, set_seed, timed
from labtts.voices import Voice

MODEL_ID = "inclusionAI/Ming-3.0-tts-0.5B"
MODEL_REVISION = "7247f4c94a5b2e987404ed28a5cfec41b1a4ca61"
CODE_COMMIT = "a39b80660bdc5a4394d5e1d41af6dbfd9c443a54"  # github.com/inclusionAI/Ming-3.0-tts
LANGUAGE = "es"


class NoRoomToSpeak(Exception):
    """The prompt alone fills the model's 512 positions: not a single latent could be generated."""


@dataclass
class MingSettings:
    """Defaults of MingTTS.synthesize(). temperature 0 makes the flow-matching sampling depend only on its noise
    (sigma) and the seed; cfg is the classifier-free guidance scale."""
    cfg: float = 2.0
    sigma: float = 0.25
    temperature: float = 0.0
    max_decode_steps: int = 1000
    split_text: bool = True
    max_length: int = 24

    @classmethod
    def from_overrides(cls, overrides: dict | None) -> "MingSettings":
        return cls(**(overrides or {}))  # an unknown key raises, so a typo in config.yaml does not pass silently


class MingTTSEngine(Engine):
    """Clones the given voice (reference audio plus its exact transcript, and the speaker embedding of the clip)."""
    supports_cloning = True

    def __init__(self, name: str, voice: Voice | None):
        super().__init__(name, voice)
        code_dir = os.environ.get("MING_TTS_CODE")
        if not code_dir:
            raise RuntimeError(f"Set MING_TTS_CODE to a clone of github.com/inclusionAI/Ming-3.0-tts at {CODE_COMMIT}")
        sys.path.insert(0, code_dir)
        from huggingface_hub import snapshot_download
        from ming_tts import MingTTS
        # The texts have no digits or symbols to expand, and the text normalizer (pynini) covers Chinese and English.
        # Encoding the reference samples random noise, so it is encoded again after each seed instead of cached:
        # a take then depends on its seed only, not on which takes ran before it.
        self.tts = MingTTS(snapshot_download(MODEL_ID, revision=MODEL_REVISION), text_normalization=False,
                           cache_prompts=False)
        self._record_stops()
        self.reference_audio, self.reference_transcript = None, None
        if voice is not None:
            self.use_reference(str(voice.reference_audio), voice.transcript)

    def _record_stops(self) -> None:
        """Wraps the decoding loop to note, per generation, whether the model's stop head ended it."""
        sample = self.tts.model.sample
        self.stopped_on_its_own = []

        def sample_recording_stop(*arguments, **keywords):
            finished, latents = False, 0
            for latent, finished in sample(*arguments, **keywords):
                latents += 1
                yield latent, finished
            self.stopped_on_its_own.append(finished)
            if not latents:
                raise NoRoomToSpeak

        self.tts.model.sample = sample_recording_stop

    def use_reference(self, reference_audio, transcript: str) -> None:
        if not isinstance(reference_audio, str):
            raise NotImplementedError(f"{self.name} clones from a file path only")
        self.reference_audio, self.reference_transcript = reference_audio, transcript

    def synthesize(self, text: str, seed: int, settings: dict | None = None) -> Synthesis:
        if self.reference_audio is None:
            raise RuntimeError(f"{self.name} has no built-in Spanish voice: give it a voice to clone")
        settings = MingSettings.from_overrides(settings)
        self.stopped_on_its_own.clear()
        set_seed(seed)

        def generate():
            wave = self.tts.synthesize(
                task="MULTILINGUAL", text=text, language_id=LANGUAGE, prompt_wav_path=self.reference_audio,
                prompt_text=self.reference_transcript, use_spk_emb=True, cfg=settings.cfg, sigma=settings.sigma,
                temperature=settings.temperature, max_length=settings.max_length,
                max_decode_steps=settings.max_decode_steps, raw_text=not settings.split_text)
            return wave[0], self.tts.sample_rate

        try:
            wave, sample_rate, seconds = timed(generate)
        except NoRoomToSpeak:  # a tenth of a second of silence keeps the take in the tables
            return Synthesis(wave=np.zeros(self.tts.sample_rate // 10, dtype=np.float32),
                             sample_rate=self.tts.sample_rate, generation_seconds=0.0, hit_token_limit=True)
        return Synthesis(wave=wave, sample_rate=sample_rate, generation_seconds=seconds,
                         hit_token_limit=not all(self.stopped_on_its_own))

    def close(self) -> None:
        del self.tts
        super().close()
