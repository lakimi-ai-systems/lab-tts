"""Acoustic measurements: pitch, loudness, brightness and speaking rate, globally and over time windows.

Everything is computed on 16 kHz mono audio. Pitch comes from pYIN; brightness is the spectral centroid of voiced
frames only (vowels and voiced consonants), so hissing fricatives such as "s" do not dominate it.
"""
import re
from dataclasses import asdict, dataclass

import librosa
import numpy as np

ANALYSIS_SAMPLE_RATE = 16000
FRAME_LENGTH = 1024  # 64 ms: about four periods of the lowest pitch tracked
HOP_LENGTH = 256  # 16 ms
PITCH_FLOOR_HZ = 60
PITCH_CEILING_HZ = 500
SILENCE_DBFS = -45


def _to_analysis_rate(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    wave = wave.astype(np.float32)
    return librosa.resample(wave, orig_sr=sample_rate, target_sr=ANALYSIS_SAMPLE_RATE) if sample_rate != ANALYSIS_SAMPLE_RATE else wave


def semitones_between(frequency_hz: float, reference_hz: float) -> float:
    return float(12 * np.log2(frequency_hz / reference_hz))


def loudness_dbfs(wave: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(wave))) + 1e-9))


@dataclass
class FrameAnalysis:
    """Per-frame pitch (NaN when unvoiced) and spectral centroid, every HOP_LENGTH samples at 16 kHz."""
    pitch_hz: np.ndarray
    centroid_hz: np.ndarray
    wave: np.ndarray

    @property
    def frame_seconds(self) -> float:
        return HOP_LENGTH / ANALYSIS_SAMPLE_RATE

    @property
    def voiced(self) -> np.ndarray:
        return ~np.isnan(self.pitch_hz)


def analyze_frames(wave: np.ndarray, sample_rate: int) -> FrameAnalysis:
    wave_16k = _to_analysis_rate(wave, sample_rate)
    pitch_hz, _, _ = librosa.pyin(wave_16k, fmin=PITCH_FLOOR_HZ, fmax=PITCH_CEILING_HZ, sr=ANALYSIS_SAMPLE_RATE,
                                  frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    centroid_hz = librosa.feature.spectral_centroid(y=wave_16k, sr=ANALYSIS_SAMPLE_RATE, n_fft=FRAME_LENGTH,
                                                    hop_length=HOP_LENGTH)[0]
    frames = min(len(pitch_hz), len(centroid_hz))
    return FrameAnalysis(pitch_hz=pitch_hz[:frames], centroid_hz=centroid_hz[:frames], wave=wave_16k)


@dataclass
class VoiceProfile:
    pitch_median_hz: float
    pitch_p95_hz: float
    pitch_spread_semitones: float  # standard deviation of voiced pitch, in semitones around the median
    centroid_median_hz: float
    centroid_p95_hz: float
    loudness_dbfs: float

    def as_dict(self) -> dict:
        return {key: round(value, 2) for key, value in asdict(self).items()}


def voice_profile(analysis: FrameAnalysis) -> VoiceProfile:
    voiced_pitch = analysis.pitch_hz[analysis.voiced]
    voiced_centroid = analysis.centroid_hz[analysis.voiced]
    median_pitch = float(np.median(voiced_pitch))
    return VoiceProfile(pitch_median_hz=median_pitch, pitch_p95_hz=float(np.percentile(voiced_pitch, 95)),
                        pitch_spread_semitones=float(np.std(12 * np.log2(voiced_pitch / median_pitch))),
                        centroid_median_hz=float(np.median(voiced_centroid)),
                        centroid_p95_hz=float(np.percentile(voiced_centroid, 95)),
                        loudness_dbfs=loudness_dbfs(analysis.wave))


def count_syllables(word: str) -> int:
    """Spanish syllables approximated as vowel groups (diphthongs count once; a few hiatuses are undercounted)."""
    return max(1, len(re.findall(r"[aeiouáéíóúü]+", word.lower())))


def windows_over_time(analysis: FrameAnalysis, window_seconds: float, words: list[str],
                      word_start_seconds: list[float]) -> list[dict]:
    """Pitch, loudness and speaking rate per window, to see whether a long generation drifts."""
    total_seconds = len(analysis.wave) / ANALYSIS_SAMPLE_RATE
    frames_per_window = int(window_seconds / analysis.frame_seconds)
    windows = []
    for window_index, window_start in enumerate(np.arange(0, total_seconds, window_seconds)):
        window_end = min(window_start + window_seconds, total_seconds)
        if window_end - window_start < 3:
            continue
        frame_slice = slice(window_index * frames_per_window, (window_index + 1) * frames_per_window)
        window_pitch = analysis.pitch_hz[frame_slice]
        window_pitch = window_pitch[~np.isnan(window_pitch)]
        samples = analysis.wave[int(window_start * ANALYSIS_SAMPLE_RATE): int(window_end * ANALYSIS_SAMPLE_RATE)]
        syllables = sum(count_syllables(word) for word, start in zip(words, word_start_seconds)
                        if window_start <= start < window_end)
        windows.append({"start_s": round(float(window_start), 1), "end_s": round(float(window_end), 1),
                        "pitch_median_hz": round(float(np.median(window_pitch)), 1) if len(window_pitch) else None,
                        "loudness_dbfs": round(loudness_dbfs(samples), 1),
                        "syllables_per_second": round(syllables / (window_end - window_start), 2)})
    return windows


@dataclass
class ShrillnessThresholds:
    """A one-second window is "shrill" when, compared with the reference voice, its median pitch rises by more than
    pitch_margin_semitones or its median brightness exceeds the reference's 95th percentile by centroid_margin.
    Windows with less than min_voiced_share voiced frames (pauses, breaths) are skipped."""
    window_seconds: float = 1.0
    pitch_margin_semitones: float = 4.0
    centroid_margin: float = 1.2
    min_voiced_share: float = 0.3


def shrill_windows(analysis: FrameAnalysis, reference: VoiceProfile, thresholds: ShrillnessThresholds) -> dict:
    frames_per_window = int(thresholds.window_seconds / analysis.frame_seconds)
    flagged_starts, voiced_windows = [], 0
    for first_frame in range(0, len(analysis.pitch_hz) - frames_per_window + 1, frames_per_window):
        window_voiced = analysis.voiced[first_frame: first_frame + frames_per_window]
        if window_voiced.mean() < thresholds.min_voiced_share:
            continue
        voiced_windows += 1
        window_pitch = np.median(analysis.pitch_hz[first_frame: first_frame + frames_per_window][window_voiced])
        window_centroid = np.median(analysis.centroid_hz[first_frame: first_frame + frames_per_window][window_voiced])
        if (semitones_between(window_pitch, reference.pitch_median_hz) > thresholds.pitch_margin_semitones
                or window_centroid > reference.centroid_p95_hz * thresholds.centroid_margin):
            flagged_starts.append(round(first_frame * analysis.frame_seconds, 1))
    return {"voiced_windows": voiced_windows, "shrill_windows": len(flagged_starts),
            "shrill_share": round(len(flagged_starts) / max(1, voiced_windows), 4), "shrill_at_seconds": flagged_starts}
