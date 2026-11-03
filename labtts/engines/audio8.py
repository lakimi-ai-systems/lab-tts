"""Audio8-TTS-Realistic-Air (Edge0): a 0.6B DualAR model with a 44.1 kHz codec, zero-shot cloning or a built-in voice.

Its code ships with the checkpoint (trust_remote_code), so the revision is pinned: a later push to the model repo
cannot silently change what the experiments run. No packages beyond the core environment are needed.

Context: text, reference and generated audio share 2048 positions (~21.5 audio frames per second), so a single
generation cannot go much beyond a minute of speech. When the context runs out the generation simply stops;
Synthesis.hit_token_limit records it.
"""
from dataclasses import asdict, dataclass

import torch

from labtts.engines.base import Engine, Synthesis, set_seed, timed
from labtts.voices import Voice

MODEL_ID = "Edge0/Audio8-TTS-Realistic-Air"
MODEL_REVISION = "23c305cc013fde378edb459f538317043c993a74"


@dataclass
class Audio8Sampling:
    """Defaults from the model card's usage example. The same temperature, top-p and top-k apply to the semantic
    token and to the acoustic codebooks. max_new_tokens is further capped by the free context."""
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 2048

    @classmethod
    def from_overrides(cls, overrides: dict | None) -> "Audio8Sampling":
        return cls(**(overrides or {}))  # an unknown key raises, so a typo in config.yaml does not pass silently


class Audio8Engine(Engine):
    """Clones the given voice; a variant with `reference: none` speaks with the model's built-in voice instead.

    The reference is encoded into codec frames once and reused for every generation.
    """
    supports_cloning = True

    def __init__(self, name: str, voice: Voice | None):
        super().__init__(name, voice)
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
                                               dtype=torch.bfloat16).eval().to("cuda")
        self.reference_codes, self.reference_transcript = None, None
        if voice is not None:
            self.use_reference(str(voice.reference_audio), voice.transcript)

    @torch.inference_mode()
    def use_reference(self, reference_audio, transcript: str) -> None:
        encoded = self.processor(text=["-"], reference_text=[transcript], reference_audio=[reference_audio],
                                 return_tensors="pt")
        codes, lengths = self.model.encode_audio(encoded["reference_audio_values"].to("cuda"),
                                                 encoded["reference_audio_lengths"].to("cuda"))
        self.reference_codes = codes[0, :, : int(lengths[0])].cpu()
        self.reference_transcript = transcript

    @torch.inference_mode()
    def synthesize(self, text: str, seed: int, settings: dict | None = None) -> Synthesis:
        settings = dict(settings or {})
        use_built_in_voice = settings.pop("reference", None) == "none" or self.reference_codes is None
        sampling = Audio8Sampling.from_overrides(settings)
        reference = {} if use_built_in_voice else {"reference_text": [self.reference_transcript],
                                                   "reference_codes": [self.reference_codes]}
        inputs = {key: value.to("cuda") for key, value in
                  self.processor(text=[text], return_tensors="pt", **reference).items()}
        set_seed(seed)
        finished = {}

        def generate():
            output = self.model.generate(**inputs, do_sample=True, return_dict_in_generate=True, **asdict(sampling))
            finished["value"] = bool(output.finished[0])
            waves, lengths = self.model.decode_audio(output.codes)
            return waves[0, : int(lengths[0])], self.model.config.codec_sample_rate

        wave, sample_rate, seconds = timed(generate)
        return Synthesis(wave=wave, sample_rate=sample_rate, generation_seconds=seconds,
                         hit_token_limit=not finished["value"])

    def close(self) -> None:
        del self.model
        super().close()
