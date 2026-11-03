# 05 · One everyday sentence, every engine

**Question.** How do the engines compare on a plain, everyday sentence, the one used in the samples posted on X? In
particular, where does Audio8-TTS-Realistic-Air's "out of breath" feel come from?

**Setup.** The ~15 s sentence in [`texts/everyday.txt`](../../texts/everyday.txt), four seeds per engine and variant
([`config.yaml`](config.yaml)):

- `qwen3-tts-0.6b` cloning the designed `warm_male` voice;
- `audio8-air` with its built-in voice (as in the post) and cloning the same `warm_male` voice, for a like-for-like
  comparison.

Every take is judged for pronunciation and for **pauses**: stretches of at least 0.2 s more than 20 dB below the
speech level, whether silence or breath. Seed 0 of each variant gets a waveform video
(`results/05-everyday-sentence/<engine>/<variant>_seed0.mp4`, made with [`tools/waveform_video.sh`](../../tools)).

**How to run.** `uv run python experiments/05-everyday-sentence/run.py [--engines name,...] [--rescore]` (needs the
voices from experiment 01; `ffmpeg` for the videos).

<!-- results:start -->
### By engine and variant (mean over seeds)

| engine | variant | voice | takes | duration (s) | pause time (s) | pause share | longest pause (s) | phonetic error | flagged words | Whisper WER | real-time factor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-tts-0.6b | default | qwen3-tts-1.7b-voicedesign/warm_male | 4 | 15.2 | 2.6 | 17% | 0.6 | 0.0036 | — | 0.0 | 1.01 |
| audio8-air | built_in_voice | built-in | 4 | 17.0 | 3.4 | 20% | 0.78 | 0.0072 | Hola,→o[∅], Hola,→o[∅] | 0.0 | 1.0 |
| audio8-air | cloned | qwen3-tts-1.7b-voicedesign/warm_male | 4 | 19.3 | 5.2 | 27% | 1.06 | 0.0 | — | 0.0 | 0.97 |

### Every take

| engine | variant | seed | duration (s) | pauses | pause time (s) | phonetic error | Whisper heard |
|---|---|---|---|---|---|---|---|
| qwen3-tts-0.6b | default | 0 | 17.09 | 9 | 4.3 | 0.0 | Hola, buenos días. Gracias por llamar. Le cuento. La reunión del jueves se ha movido al viernes a las 10 de la mañana, en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| qwen3-tts-0.6b | default | 1 | 15.57 | 6 | 3.02 | 0.0096 | Hola, buenos días. Gracias por llamar. Le cuento, la reunión del jueves se ha movido al viernes a las 10 de la mañana, en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| qwen3-tts-0.6b | default | 2 | 14.13 | 4 | 2.05 | 0.0048 | Hola, buenos días. Gracias por llamar. Le cuento, la reunión del jueves se ha movido al viernes a las 10 de la mañana en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| qwen3-tts-0.6b | default | 3 | 14.21 | 5 | 1.23 | 0.0 | Hola, buenos días. Gracias por llamar. Le cuento. La reunión del jueves se ha movido al viernes a las 10 de la mañana en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| audio8-air | built_in_voice | 0 | 17.51 | 7 | 3.57 | 0.0096 | Hola, buenos días, gracias por llamar. Le cuento. La reunión del jueves se ha movido al viernes a las 10 de la mañana en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| audio8-air | built_in_voice | 1 | 18.11 | 10 | 4.02 | 0.0 | Hola, buenos días, gracias por llamar. Le cuento, la reunión del jueves se ha movido al viernes a las 10 de la mañana en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| audio8-air | built_in_voice | 2 | 15.56 | 6 | 2.72 | 0.0048 | Hola, buenos días, gracias por llamar. Le cuento, la reunión del jueves se ha movido al viernes a las 10 de la mañana, en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| audio8-air | built_in_voice | 3 | 16.95 | 6 | 3.17 | 0.0144 | Hola buenos días, gracias por llamar. Le cuento, la reunión del jueves se ha movido al viernes a las 10 de la mañana, en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| audio8-air | cloned | 0 | 19.46 | 9 | 5.7 | 0.0 | Hola, buenos días. Gracias por llamar. Le cuento, la reunión del jueves se ha movido al viernes a las 10 de la mañana en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| audio8-air | cloned | 1 | 18.81 | 7 | 4.86 | 0.0 | Hola, buenos días, gracias por llamar. Le cuento, la reunión del jueves se ha movido al viernes a las 10 de la mañana en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| audio8-air | cloned | 2 | 19.92 | 7 | 5.28 | 0.0 | Hola, buenos días, gracias por llamar. Le cuento, la reunión del jueves se ha movido al viernes a las 10 de la mañana en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
| audio8-air | cloned | 3 | 19.18 | 8 | 5.04 | 0.0 | Hola. Buenos días, gracias por llamar. Le cuento, la reunión del jueves se ha movido al viernes a las 10 de la mañana en la sala grande de la segunda planta. Si no puede asistir, avíseme antes del miércoles y buscamos otra fecha que le venga mejor. Que tenga un buen día. |
<!-- results:end -->

## Conclusion

- **All three read the sentence correctly.** Whisper hears every word in every take (WER 0). Audio8 with its
  built-in voice swallows part of the opening *Hola* in two of four takes; cloning the designed voice it is
  flawless on this sentence (phonetic error 0), and Qwen3-TTS 0.6B is close (0.0036).
- **Audio8's "out of breath" feel is mostly time spent pausing.** Cloning the very same voice, Audio8 takes 19.3 s
  against Qwen's 15.2 s and spends **twice as long in pauses** (5.2 s against 2.6 s, 27 % of the take against 17 %),
  with longer pauses too (up to 1.1 s against 0.6 s). With its built-in voice the gap is smaller (17.0 s, 3.4 s of
  pauses) but still there. Those long, breathy gaps between phrases are what the ear reads as a speaker short of
  breath.

## Limitations

- **The breaths themselves are not measured.** We tried to detect them acoustically (quiet unvoiced stretches with a
  dull spectrum, and voice-quality measures such as harmonic-to-noise ratio). It did not separate the two models:
  Qwen's pauses also hold low-level breath-like sounds. What the numbers show reliably is the time spent in pauses;
  whether a pause carries an audible breath is still a matter of listening.
- Four seeds and a single sentence: enough to show the pause gap consistently (every Audio8 cloned take pauses
  longer than every Qwen take), not to characterize all styles of speech.
