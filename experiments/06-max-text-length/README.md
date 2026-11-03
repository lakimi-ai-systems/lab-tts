# 06 · How much text fits in one generation?

**Question.** How long a text can each engine read in a single call before it stops early, cuts off or runs into its
length cap?

**Setup.** Each take reads the first 1, 2, 4, 8, 12, 16 or all 22 sentences of
[`texts/long_form_ai.txt`](../../texts/long_form_ai.txt) in one generation, with two seeds per length
([`config.yaml`](config.yaml)). A take counts as **read in full** when Whisper's WER against the requested text is at
most 0.1. *Hit length cap* counts takes that ended because the engine ran out of room, rather than on their own.

**How to run.** `uv run python experiments/06-max-text-length/run.py [--engines name,...] [--rescore]` (needs the
voices from experiment 01).

<!-- results:start -->
| engine | variant | voice | sentences | words | read in full | duration (s) | Whisper WER | hit length cap |
|---|---|---|---|---|---|---|---|---|
| qwen3-tts-0.6b | default | qwen3-tts-1.7b-voicedesign/warm_male | 1 | 12 | 2/2 | 4.4 | 0.0 | 0 |
| qwen3-tts-0.6b | default | qwen3-tts-1.7b-voicedesign/warm_male | 2 | 42 | 2/2 | 12.4 | 0.0 | 0 |
| qwen3-tts-0.6b | default | qwen3-tts-1.7b-voicedesign/warm_male | 4 | 68 | 2/2 | 22.1 | 0.0 | 0 |
| qwen3-tts-0.6b | default | qwen3-tts-1.7b-voicedesign/warm_male | 8 | 153 | 2/2 | 51.9 | 0.003 | 0 |
| qwen3-tts-0.6b | default | qwen3-tts-1.7b-voicedesign/warm_male | 12 | 201 | 2/2 | 71.6 | 0.0 | 0 |
| qwen3-tts-0.6b | default | qwen3-tts-1.7b-voicedesign/warm_male | 16 | 278 | 2/2 | 97.8 | 0.002 | 0 |
| qwen3-tts-0.6b | default | qwen3-tts-1.7b-voicedesign/warm_male | 22 | 375 | 2/2 | 129.2 | 0.0 | 0 |
| audio8-air | cloned | qwen3-tts-1.7b-voicedesign/warm_male | 1 | 12 | 2/2 | 4.8 | 0.0 | 0 |
| audio8-air | cloned | qwen3-tts-1.7b-voicedesign/warm_male | 2 | 42 | 2/2 | 15.1 | 0.0 | 0 |
| audio8-air | cloned | qwen3-tts-1.7b-voicedesign/warm_male | 4 | 68 | 2/2 | 23.9 | 0.007 | 0 |
| audio8-air | cloned | qwen3-tts-1.7b-voicedesign/warm_male | 8 | 153 | 0/2 | 28.6 | 0.663 | 0 |
| audio8-air | cloned | qwen3-tts-1.7b-voicedesign/warm_male | 12 | 201 | 0/2 | 1.8 | 0.985 | 0 |
| audio8-air | cloned | qwen3-tts-1.7b-voicedesign/warm_male | 16 | 278 | 0/2 | 5.3 | 0.98 | 0 |
| audio8-air | cloned | qwen3-tts-1.7b-voicedesign/warm_male | 22 | 375 | 0/2 | 2.4 | 0.985 | 0 |
<!-- results:end -->

## Conclusion

- **Qwen3-TTS 0.6B reads the whole two minutes in one go.** Every length up to the full 375-word text (129 s of
  speech) is read in full, with WER at or near 0.
- **Audio8-TTS-Realistic-Air breaks at around 70 words.** It reads up to four sentences (68 words, ~24 s of speech)
  correctly. At eight sentences (153 words) it starts well and then derails into speech that no longer follows the
  text (WER 0.66). From twelve sentences on, it produces only one to six seconds of garbled speech and stops.
- **It is not the context running out.** No Audio8 take ran into the length cap: the model ends the generation, or
  goes off script, on its own, well before its 2048 positions are used. Longer inputs are probably simply outside
  what it was trained on.
- **In practice:** with Audio8, keep each call to a sentence or two (under ~70 words or ~25 s of speech) and join
  the audio. Qwen can take whole paragraphs, which also sounds more natural (see [experiment 03](../03-long-form)).

## Limitations

- Two seeds per length, and lengths in steps (4, then 8 sentences): the break lies somewhere between 68 and 153
  words, and may move with the text and the voice.
- "Read in full" uses Whisper's WER with a 0.1 threshold; it says nothing about pronunciation quality, which
  [experiment 02](../02-phonetics) covers.
