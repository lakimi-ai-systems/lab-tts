# 04 · Why does a cloned voice turn shrill?

**Question.** A voice designed with VoiceDesign sounds right, but when it is cloned and used to generate a lot of
speech it sometimes turns shrill. What causes it?

**Suspects.**

1. **The reference.** A lively, high-pitched reference may push the clone up.
2. **Talker sampling.** The talker picks one semantic token per frame (content and prosody); a high temperature may
   produce pitch excursions.
3. **Subtalker sampling.** The subtalker picks the remaining acoustic codebooks of each frame (fine timbre); a high
   temperature may make the timbre brighter or harsher.
4. **Re-cloning.** Using a generated output as the next reference (a copy of a copy) may accumulate drift.

**Setup.** Six ~15 s paragraphs ([`texts/paragraphs.jsonl`](../../texts/paragraphs.jsonl)) for every combination
of reference (calm `warm_male`, lively `warm_male_energetic`) and sampling variant, with paragraph *i* always on seed
*i*. The variants change one setting at a time from the defaults (temperature 0.9 for both talker and subtalker):
`talker_cold` / `talker_hot` set the talker temperature to 0.6 / 1.2, and `subtalker_cold` / `subtalker_hot` do the
same for the subtalker. Plus a re-cloning chain of six steps starting from the calm voice.

**How to run.** `uv run python experiments/04-shrill-voice/run.py [--engines name,...] [--rescore]` (needs the
voices from experiment 01; engines that cannot clone are skipped).

**Measuring "shrill".** Each generation is cut into one-second windows. A voiced window counts as shrill when, compared
with the calm target voice, its median pitch is more than 4 semitones higher or its median brightness (spectral
centroid of voiced frames) is more than 20 % above the target's 95th percentile. This is an acoustic proxy, not a
listening test: it flags candidates to listen to, and the thresholds are in [`config.yaml`](config.yaml).

<!-- results:start -->
### Reference voices

| voice | pitch median (Hz) | pitch p95 (Hz) | pitch spread (st) | brightness median (Hz) |
|---|---|---|---|---|
| qwen3-tts-1.7b-voicedesign/warm_male | 144.36 | 178.04 | 2.93 | 1365.02 |
| qwen3-tts-1.7b-voicedesign/warm_male_energetic | 263.24 | 324.37 | 2.15 | 1736.9 |

### Reference × variant (6 paragraphs each)

| engine | reference | variant | shrill windows (mean share) | generations with shrill windows | pitch median vs target (st) | pitch p95 vs target (st) | brightness vs target | pitch median vs own reference (st) | brightness vs own reference | phonetic error |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male | default | 0.0% | 0/6 | -2.13 | -0.83 | 0.842 | -2.13 | 0.842 | 0.0034 |
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male | talker_cold | 0.0% | 0/6 | -0.79 | -0.38 | 0.864 | -0.79 | 0.864 | 0.0026 |
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male | talker_hot | 9.8% | 4/6 | -1.12 | 1.22 | 0.839 | -1.12 | 0.839 | 0.0044 |
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male | subtalker_cold | 2.6% | 1/6 | -1.45 | -0.68 | 0.834 | -1.45 | 0.834 | 0.0033 |
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male | subtalker_hot | 2.4% | 1/6 | -0.65 | 1.06 | 0.852 | -0.65 | 0.852 | 0.0033 |
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male_energetic | default | 98.6% | 6/6 | 9.87 | 9.75 | 1.278 | -0.53 | 1.004 | 0.0049 |
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male_energetic | talker_cold | 100.0% | 6/6 | 10.37 | 10.08 | 1.303 | -0.03 | 1.024 | 0.0043 |
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male_energetic | talker_hot | 98.6% | 6/6 | 10.12 | 10.02 | 1.289 | -0.28 | 1.014 | 0.0068 |
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male_energetic | subtalker_cold | 100.0% | 6/6 | 10.08 | 10.29 | 1.308 | -0.32 | 1.028 | 0.0076 |
| qwen3-tts-0.6b | qwen3-tts-1.7b-voicedesign/warm_male_energetic | subtalker_hot | 98.3% | 6/6 | 9.69 | 10.27 | 1.246 | -0.71 | 0.98 | 0.006 |

### Re-cloning chain (each step clones the previous output)

| engine | step | shrill windows | pitch median vs target (st) | pitch p95 vs target (st) | brightness vs target | phonetic error |
|---|---|---|---|---|---|---|
| qwen3-tts-0.6b | 1 | 0.0% | -1.4 | 0.67 | 0.841 | 0.0 |
| qwen3-tts-0.6b | 2 | 0.0% | -0.8 | 0.77 | 0.791 | 0.0099 |
| qwen3-tts-0.6b | 3 | 0.0% | -1.8 | 0.35 | 0.686 | 0.0 |
| qwen3-tts-0.6b | 4 | 0.0% | -2.4 | 0.17 | 0.71 | 0.0 |
| qwen3-tts-0.6b | 5 | 0.0% | -3.8 | -1.76 | 0.6 | 0.0 |
| qwen3-tts-0.6b | 6 | 0.0% | -2.95 | -2.13 | 0.595 | 0.0 |
<!-- results:end -->

## Conclusion (qwen3-tts-0.6b)

- **The reference sets the pitch, and the clone copies it faithfully.** Cloned from the lively reference, every
  generation sits about 10 semitones above the calm voice, almost all of the time and whatever the sampling. Compared
  with its *own* reference it is within 0.7 semitones and the same brightness: the model does not exaggerate the
  reference, it reproduces it. A clone that is shrill all the time has a shrill reference. Check the reference first.
- **A hot talker produces shrill moments.** With the calm reference, the default settings and a cooler talker
  (0.6) never crossed the threshold. Raising the talker temperature to 1.2 produced shrill windows in four of the
  six paragraphs (9.8 % of the speech), and the high end of the pitch range (95th percentile) rose by about
  2 semitones. That matches "fine most of the time, shrill now and then". Keep the talker temperature at the default
  0.9 or lower: 0.6 removed those moments with no loss of pronunciation (phonetic error 0.0026, the best of the grid).
- **The subtalker is not the lever.** Making it hotter or colder left one paragraph out of six with a couple of
  flagged windows (≈ 2.5 %) in both directions: at noise level, with no clear trend.
- **Re-cloning degrades the voice, but it does not make it shrill.** Each copy of a copy gets darker and lower: by the
  sixth step, brightness has dropped to 60 % of the calm voice and the median pitch is about 3 semitones down. Always
  clone from the original reference, with the prompt built once.

**Recipe against shrillness:** pick a calm reference take (experiment 01 selects the one with the lowest pitch spread),
build the clone prompt once from the original file with its exact transcript, keep the talker temperature at 0.9 or
below, and never use generated audio as the next reference.

## Limitations

- Six paragraphs per cell: the ranking between variants is clear, the exact percentages are not.
- "Shrill" is an acoustic proxy (pitch and brightness against the calm voice), not a listening test. Listen to the
  flagged windows (`shrill_at_seconds` in `results/04-shrill-voice/results.json`, after a run) before drawing fine
  conclusions.
- A cloned paragraph sits about 2 semitones below the reference even at default settings, probably because narrative
  text is read lower than the reference's greeting. The comparisons between variants are unaffected, since they all
  use the same texts and seeds.
