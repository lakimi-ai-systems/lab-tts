# 01 · Voice design

**Question.** Can we get a good, reusable Castilian Spanish reference voice without recording anyone?

**Setup.** Qwen3-TTS 1.7B VoiceDesign reads the same ~10 s sentence from a natural-language description
([`config.yaml`](config.yaml)), with four seeds and the description written in English and in Spanish. Every take is
scored automatically (Whisper WER, phonetic judge, pitch profile) and the best one is saved under
[`voices/`](../../voices) for the other experiments:

- `warm_male`: among the takes with no recognition error and no flagged word, the **calmest** one (lowest pitch
  spread, the standard deviation of pitch in semitones).
- `warm_male_energetic`: the same man described as euphoric and high-pitched, used as the "bad reference" in
  [experiment 04](../04-shrill-voice).

**How to run.** `uv run python experiments/01-voice-design/run.py` (overwrites `voices/`).

<!-- results:start -->
| voice | description | seed | duration (s) | WER | phonetic error | flagged words | pitch median (Hz) | pitch spread (st) |  |
|---|---|---|---|---|---|---|---|---|---|
| warm_male | english | 0 | 10.86 | 0.0 | 0.0 | — | 144.36 | 2.93 | **chosen** |
| warm_male | english | 1 | 11.02 | 0.0 | 0.0 | — | 162.98 | 3.68 |  |
| warm_male | english | 2 | 10.38 | 0.0 | 0.0 | — | 94.15 | 3.53 |  |
| warm_male | english | 3 | 8.38 | 0.0 | 0.0092 | — | 199.5 | 3.19 |  |
| warm_male | spanish | 0 | 10.22 | 0.0 | 0.0 | — | 148.59 | 4.49 |  |
| warm_male | spanish | 1 | 8.86 | 0.0 | 0.0 | — | 128.24 | 4.27 |  |
| warm_male | spanish | 2 | 9.34 | 0.0 | 0.0092 | y,→[y] | 122.1 | 4.44 |  |
| warm_male | spanish | 3 | 11.5 | 0.0 | 0.0 | — | 198.35 | 4.85 |  |
| warm_male_energetic | english | 0 | 7.66 | 0.0 | 0.055 | buenas→[∅]u[∅]nas, gracias→[k]a[k]as | 324.08 | 3.81 |  |
| warm_male_energetic | english | 1 | 8.06 | 0.0 | 0.0 | — | 263.24 | 2.15 | **chosen** |
| warm_male_energetic | english | 2 | 12.06 | 0.0 | 0.0367 | Hola,→[∅], a→[∅] | 419.07 | 2.03 |  |
| warm_male_energetic | english | 3 | 10.22 | 0.0 | 0.0183 | Hola,→[∅]a | 285.41 | 3.49 |  |
<!-- results:end -->

## Conclusion

- **Yes: a usable reference voice comes out of a text description.** All eight takes of the calm description are
  fully intelligible (WER 0) and almost perfectly pronounced (phonetic error ≤ 0.009).
- **Generate several takes: the same description gives very different voices.** Depending on the seed, the median
  pitch of the "medium-deep" man goes from 94 Hz to 199 Hz. Treat the description as a starting point and choose.
- **The English description gave calmer takes.** Every English take has a lower pitch spread (2.9–3.7 semitones)
  than every Spanish one (4.3–4.9), with the same content. The accent was asked for explicitly in both
  ("Native Castilian Spanish accent from Spain" / "Castellano peninsular de España").
- **Excited voices are harder to get right.** Three of the four euphoric takes have mispronounced or swallowed words,
  against one minor flag in eight calm takes.

## Limitations

- Four seeds per description: enough to see the spread, not to rank descriptions in general.
- "Calm" is approximated by pitch spread. It is a reasonable proxy for an even, reassuring delivery, but the final
  pick should still be checked by ear.
- The English-versus-Spanish gap is shown for this one description. It matches what we saw while preparing the repo,
  but other descriptions may behave differently.
