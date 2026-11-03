# 03 · Long-form speech: one shot or sentence by sentence?

**Question.** Does the voice hold up over two minutes of speech, and does it sound better generated in a single
call or one sentence at a time (the way a voice gateway usually feeds a TTS)?

**Setup.** A ~375-word text about AI ([`texts/long_form_ai.txt`](../../texts/long_form_ai.txt)), two seeds, both
modes. In `per_sentence` mode each of the 22 sentences is a separate call with the same clone prompt, joined with
0.25 s of silence. Pronunciation is judged on the whole audio; pitch, loudness and speaking rate (syllables per
second, from the word timings of the phonetic judge's recognizer) are tracked every 20 s.

**How to run.** `uv run python experiments/03-long-form/run.py [--engines name,...] [--rescore]` (needs the voices
from experiment 01).

<!-- results:start -->
### Pronunciation and drift

| engine | variant | mode | seed | duration (s) | phonetic error | flagged words | Whisper WER | loudness last − first window (dB) | window pitch range (Hz) |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-tts-0.6b | default | one_shot | 0 | 128.9 | 0.0011 | — | 0.0 | -3.2 | 125.0–146.9 |
| qwen3-tts-0.6b | default | per_sentence | 0 | 134.1 | 0.0023 | — | 0.0 | -0.4 | 124.2–141.9 |
| qwen3-tts-0.6b | default | one_shot | 1 | 129.6 | 0.0028 | — | 0.0 | -4.9 | 123.5–158.3 |
| qwen3-tts-0.6b | default | per_sentence | 1 | 132.3 | 0.0034 | cantidad→kantidad[de], de→[∅]e | 0.0 | 1.6 | 125.7–144.4 |
| audio8-air | cloned | one_shot | 0 | 2.8 | 1.0272 | Hace→[∅], apenas→, unos→o[∅], años,→, hablar→, con→, una→, máquina→ … and 365 more | 0.984 | — | — |
| audio8-air | cloned | per_sentence | 0 | 159.5 | 0.0011 | — | 0.0 | 0.1 | 134.7–152.1 |
| audio8-air | cloned | one_shot | 1 | 2.0 | 1.0034 | Hace→[oik]e, apenas→[∅]p[∅]as, unos→[i], años,→a[d], hablar→, con→, una→a, máquina→m[u] … and 366 more | 0.9867 | — | — |
| audio8-air | cloned | per_sentence | 1 | 163.1 | 0.0034 | siguiente.→sigiente[oa] | 0.0 | -0.6 | 134.7–147.7 |

### Mean over seeds

| engine | variant | mode | phonetic error | loudness drift (dB) |
|---|---|---|---|---|
| qwen3-tts-0.6b | default | one_shot | 0.0019 | -4.1 |
| qwen3-tts-0.6b | default | per_sentence | 0.0029 | 0.6 |
| audio8-air | cloned | one_shot | 1.0153 | — |
| audio8-air | cloned | per_sentence | 0.0022 | -0.2 |

### Every 20 s · qwen3-tts-0.6b, default, one_shot, seed 0

| window | pitch median (Hz) | loudness (dBFS) | syllables / s |
|---|---|---|---|
| 0–20 s | 126.4 | -29.3 | 6.1 |
| 20–40 s | 125.0 | -30.4 | 5.95 |
| 40–60 s | 125.0 | -32.1 | 5.5 |
| 60–80 s | 146.9 | -31.2 | 5.3 |
| 80–100 s | 140.3 | -30.7 | 6.1 |
| 100–120 s | 141.1 | -31.3 | 5.8 |
| 120–129 s | 131.6 | -32.5 | 6.06 |

### Every 20 s · qwen3-tts-0.6b, default, per_sentence, seed 0

| window | pitch median (Hz) | loudness (dBFS) | syllables / s |
|---|---|---|---|
| 0–20 s | 124.2 | -29.1 | 5.65 |
| 20–40 s | 141.9 | -26.1 | 5.9 |
| 40–60 s | 129.4 | -29.4 | 5.35 |
| 60–80 s | 134.7 | -29.3 | 5.15 |
| 80–100 s | 136.3 | -28.5 | 5.85 |
| 100–120 s | 138.6 | -28.8 | 5.4 |
| 120–134 s | 126.4 | -29.5 | 5.96 |

### Every 20 s · audio8-air, cloned, one_shot, seed 0

| window | pitch median (Hz) | loudness (dBFS) | syllables / s |
|---|---|---|---|

### Every 20 s · audio8-air, cloned, per_sentence, seed 0

| window | pitch median (Hz) | loudness (dBFS) | syllables / s |
|---|---|---|---|
| 0–20 s | 147.7 | -26.8 | 5.25 |
| 20–40 s | 134.7 | -27.3 | 4.9 |
| 40–60 s | 144.4 | -28.2 | 4.75 |
| 60–80 s | 145.6 | -28.6 | 3.9 |
| 80–100 s | 152.1 | -27.0 | 4.55 |
| 100–120 s | 142.7 | -27.2 | 4.7 |
| 120–140 s | 137.8 | -27.2 | 4.8 |
| 140–160 s | 135.5 | -26.7 | 4.88 |
<!-- results:end -->

## Conclusion (qwen3-tts-0.6b)

- **The voice holds up for two minutes in both modes.** Every take is fully intelligible (Whisper WER 0 on 375
  words), with near-perfect pronunciation (phonetic error 0.001–0.003). There is no runaway, repetition or timbre
  collapse in a single 2-minute generation, and the speaking rate stays steady at about 5–6 syllables per second.
- **One shot slowly loses loudness.** Over two minutes a single generation gets 3–5 dB quieter (−4.1 dB on average),
  a gradual fade that can come across as the speaker tiring. Sentence by sentence stays level (+0.6 dB), because every
  sentence starts again from the reference.
- **One shot sounds more natural.** By ear, both one-shot takes sound very good and consistent from start to end.
  Sentence by sentence, intonation varies a lot from one sentence to the next, because every sentence restarts its
  prosody from scratch. One shot's pronunciation is also slightly better (0.0019 against 0.0029).
- **In practice: feed the model several sentences at a time.** Sending one sentence per call, as many voice gateways
  do, gives up that continuity. Grouping a few sentences per call (a few seconds to a few tens of seconds of speech)
  keeps the natural flow without reaching the lengths where the fade becomes audible. If you do generate minutes in
  one go, normalize loudness afterwards.

## Conclusion (audio8-air)

- **It cannot read the whole text in one go.** Both one-shot takes stop after 2–3 seconds of garbled speech
  (Whisper WER ≈ 0.98). The model ends the generation itself, well before its 2048-position context runs out.
  [Experiment 06](../06-max-text-length) finds where it breaks: about 70 words.
- **Sentence by sentence it holds up well.** Both takes are fully intelligible (WER 0) with very good pronunciation
  (phonetic error 0.0022, against Qwen's 0.0029 in the same mode), and the loudness stays level.
- **It is slower and pausier.** The same text takes 160 s instead of Qwen's 132 s, at about 4–5 syllables per
  second instead of 5–6, mostly because of longer pauses (measured in [experiment 05](../05-everyday-sentence)).

## Limitations

- Two seeds per mode. The loudness fade appears in both one-shot takes (−3.2 and −4.9 dB), but its size will vary.
- The preference for one shot comes from listening (one listener, all four takes), not from a metric. The pitch
  windows do not capture it: sentence-to-sentence intonation changes average out over 20 s.
- Window medians smooth over short events: a single odd word would not show in these tables (the phonetic judge
  covers that).
