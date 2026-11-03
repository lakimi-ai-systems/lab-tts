"""Phonetic judge: flags mispronounced words that Whisper silently fixes.

Whisper decodes with a strong language model, so it "corrects" what it hears: a voice that says "confisa" is
transcribed as "confiesa" and the take scores a perfect WER. This judge uses a character-level CTC recognizer for
Spanish with greedy decoding and no language model, so it writes down what actually sounds.

Reference and transcript are mapped to an approximate phonetic spelling of Peninsular Spanish (distinguishes s/z,
yeísta), so only audible differences count: b/v, ll/y or a silent h are not errors. The two strings are aligned
character by character (the CTC model splits words poorly) and the edits are attributed back to reference words.

Known blind spots: it does not catch seseo or a lost diphthong in very frequent words, because the recognizer itself
was trained on speakers with those features.
"""
import difflib
import re
import unicodedata
from dataclasses import dataclass, field

import librosa
import numpy as np

CTC_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
# A word is flagged with at least this many edits, or above this share of edited characters
MIN_EDITS_PER_WORD = 2
MAX_RELATIVE_DIFFERENCE = 0.34
# Long audio is recognized in overlapping chunks to keep memory bounded
CHUNK_SECONDS = 20
CHUNK_OVERLAP_SECONDS = 4

# Hard g is written as a temporary "G" so that "gue" -> "Ge" is not turned into "je" by the soft-g rule after it
SPELLING_RULES = [
    ("ch", "C"), ("ll", "y"), ("rr", "R"), ("qu", "k"),
    ("güe", "Gue"), ("güi", "Gui"), ("gue", "Ge"), ("gui", "Gi"),
    ("ge", "je"), ("gi", "ji"),
    ("ce", "ze"), ("ci", "zi"), ("c", "k"),
    ("v", "b"), ("w", "u"), ("x", "ks"), ("h", ""),
]


def _strip_accents(text: str) -> str:
    """Removes accents but keeps ñ (n + combining tilde) and ü until the spelling rules have used them."""
    decomposed = unicodedata.normalize("NFD", text).replace("ñ", "ñ").replace("ü", "ü")
    return "".join(character for character in decomposed if unicodedata.category(character) != "Mn")


def to_phonetic_spelling(word: str) -> str:
    """Approximate pronunciation of a Peninsular Spanish word, e.g. "cielo" -> "zielo", "hueso" -> "ueso"."""
    word = re.sub(r"[^a-zñü]", "", _strip_accents(word.lower()))
    for spelling, sound in SPELLING_RULES:
        word = word.replace(spelling, sound)
    word = word.replace("G", "g").replace("ü", "u")
    word = re.sub(r"^r", "R", word)  # word-initial r is a trill
    return re.sub(r"y$", "i", word)  # word-final y is a vowel ("muy", "rey")


def phonetic_words(text: str) -> list[str]:
    return [spelling for spelling in (to_phonetic_spelling(word) for word in text.replace("-", " ").split()) if spelling]


@dataclass
class PhoneticVerdict:
    error_rate: float  # edited characters / characters of the reference, in phonetic spelling
    flagged_words: list[dict] = field(default_factory=list)  # {"expected", "heard", "edits"}
    transcript: str = ""
    # what the recognizer heard, word by word, with the second each word starts (cheap: comes from CTC decoding)
    heard_words: list[str] = field(default_factory=list)
    word_start_seconds: list[float] = field(default_factory=list)


class PhoneticJudge:
    def __init__(self, device: str = "cuda:0"):
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, pipeline
        # Tokenizer and feature extractor are passed explicitly: letting the pipeline load the repo's processor
        # would bring in the n-gram language model it ships with, and we would be "correcting" again
        self.recognizer = pipeline("automatic-speech-recognition", model=Wav2Vec2ForCTC.from_pretrained(CTC_MODEL),
                                   tokenizer=Wav2Vec2CTCTokenizer.from_pretrained(CTC_MODEL),
                                   feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained(CTC_MODEL),
                                   device=device, chunk_length_s=CHUNK_SECONDS, stride_length_s=CHUNK_OVERLAP_SECONDS)

    def evaluate(self, reference_text: str, wave: np.ndarray, sample_rate: int) -> PhoneticVerdict:
        wave_16k = librosa.resample(wave.astype(np.float32), orig_sr=sample_rate, target_sr=16000) if sample_rate != 16000 else wave
        output = self.recognizer({"raw": wave_16k, "sampling_rate": 16000}, return_timestamps="word")
        transcript = output["text"]
        original_words = [word for word in reference_text.replace("-", " ").split() if to_phonetic_spelling(word)]
        reference = phonetic_words(reference_text)
        reference_string, heard_string = "".join(reference), "".join(phonetic_words(transcript))
        word_of_character = [index for index, word in enumerate(reference) for _ in word]
        edits_per_word = [0] * len(reference)
        heard_per_word = [""] * len(reference)
        matcher = difflib.SequenceMatcher(a=reference_string, b=heard_string, autojunk=False)
        for operation, reference_start, reference_end, heard_start, heard_end in matcher.get_opcodes():
            if operation == "equal":
                for position in range(reference_start, reference_end):
                    heard_per_word[word_of_character[position]] += heard_string[heard_start + position - reference_start]
                continue
            # an insertion is charged to the previous word (or to the first one at the start)
            affected_words = sorted({word_of_character[position] for position in range(reference_start, reference_end)}) or \
                ([word_of_character[max(0, reference_start - 1)]] if word_of_character else [])
            if not affected_words:
                continue
            cost = max(reference_end - reference_start, heard_end - heard_start)
            for word_index in affected_words:
                edits_per_word[word_index] += max(1, round(cost / len(affected_words)))
            heard_per_word[affected_words[0]] += f"[{heard_string[heard_start:heard_end] or '∅'}]"
        flagged_words = []
        for word_index, edits in enumerate(edits_per_word):
            if edits >= MIN_EDITS_PER_WORD or edits / max(1, len(reference[word_index])) > MAX_RELATIVE_DIFFERENCE:
                flagged_words.append({"expected": original_words[word_index], "heard": heard_per_word[word_index],
                                      "edits": edits})
        return PhoneticVerdict(error_rate=sum(edits_per_word) / max(1, len(reference_string)),
                               flagged_words=flagged_words, transcript=transcript,
                               heard_words=[chunk["text"] for chunk in output["chunks"]],
                               word_start_seconds=[chunk["timestamp"][0] for chunk in output["chunks"]])
