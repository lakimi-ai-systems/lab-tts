"""Whisper transcription and word error rate: the usual intelligibility metric, kept for comparison.

Whisper is good at catching gross failures (skipped or invented words) but forgiving with mispronunciations, which it
fixes through its language model. See labtts.judges.phonetic for the stricter judge.

Two details keep WER honest:
- Whisper writes numbers as digits ("23", "12%") while the test texts spell them out, so digits and a few symbols are
  spelled out in Spanish before comparing.
- Audio longer than 30 s is transcribed with Whisper's sequential long-form decoding. Independent 30 s chunks
  hallucinate at the seams (repeated words, duplicated passages) and would be blamed on the TTS.
"""
import re
import unicodedata
from dataclasses import dataclass, field

import librosa
import numpy as np
import torch
from num2words import num2words

WHISPER_MODEL = "openai/whisper-large-v3-turbo"
SYMBOLS_SPELLED_OUT = {"%": " por ciento", "€": " euros", "GB": " gigas"}


@dataclass
class Transcript:
    text: str
    word_start_seconds: list[float] = field(default_factory=list)


def spell_out_numbers(text: str) -> str:
    """'24 GB' -> 'veinticuatro gigas', '12%' -> 'doce por ciento', '7,5' -> 'siete coma cinco'."""
    for symbol, words in SYMBOLS_SPELLED_OUT.items():
        text = text.replace(symbol, words)
    text = re.sub(r"(\d)\.(\d{3})\b", r"\1\2", text)  # thousands separator: 1.500 -> 1500
    text = re.sub(r"(\d+),(\d+)", lambda match: f"{match[1]} coma {match[2]}", text)
    return re.sub(r"\d+", lambda match: num2words(int(match[0]), lang="es"), text)


def normalize_words(text: str) -> list[str]:
    """Lowercase words with numbers spelled out, without accents or punctuation (ñ is kept: it is a different letter)."""
    decomposed = unicodedata.normalize("NFD", spell_out_numbers(text).lower()).replace("ñ", "ñ")
    without_accents = "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
    return re.sub(r"[^\w\s]", " ", without_accents).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words, hypothesis_words = normalize_words(reference), normalize_words(hypothesis)
    previous_row = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, 1):
        row = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, 1):
            row.append(min(previous_row[hypothesis_index] + 1, row[-1] + 1,
                           previous_row[hypothesis_index - 1] + (reference_word != hypothesis_word)))
        previous_row = row
    return previous_row[-1] / max(1, len(reference_words))


class WhisperJudge:
    def __init__(self, device: str = "cuda:0"):
        from transformers import pipeline
        self.recognizer = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, dtype=torch.float16, device=device)

    def transcribe(self, wave: np.ndarray, sample_rate: int, word_timestamps: bool = False) -> Transcript:
        wave_16k = librosa.resample(wave.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        long_form = len(wave_16k) > 30 * 16000
        # long-form decoding needs timestamps; without conditioning on previous text a bad window cannot cascade
        output = self.recognizer({"raw": wave_16k, "sampling_rate": 16000},
                                 return_timestamps="word" if word_timestamps else long_form,
                                 generate_kwargs={"language": "spanish", "task": "transcribe",
                                                  "condition_on_prev_tokens": False})
        starts = [chunk["timestamp"][0] or 0.0 for chunk in output.get("chunks", [])] if word_timestamps else []
        return Transcript(text=output["text"].strip(), word_start_seconds=starts)
