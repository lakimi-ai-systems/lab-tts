# 02 · Phonetics in Castilian Spanish

**Question.** How accurately does each engine pronounce Castilian Spanish, and do its sampling settings matter?

**Setup.** Eleven short texts ([`texts/phonetics.jsonl`](../../texts/phonetics.jsonl)), each aimed at a known
difficulty: numbers, z/c versus s, proper names, trilled r, jota, ll/y/ñ, long words, loanwords, questions and a long
sentence. Engines that clone use the designed `warm_male` voice. Two seeds per text, engine and variant
([`config.yaml`](config.yaml)); for `qwen3-tts-0.6b`, `default` is the model's own settings (temperature 0.9, top-k
50) and `conservative` lowers them to 0.6 and 20. Every take is judged twice: by Whisper (WER) and by the
[phonetic judge](../../labtts/judges/phonetic.py).

**How to run.** `uv run python experiments/02-phonetics/run.py [--engines name,...] [--rescore]` (needs the voices
from experiment 01).

<!-- results:start -->
### By engine and variant

| engine | variant | voice | takes | phonetic error (mean) | takes with flagged words | flagged words | Whisper WER (mean) | flagged but WER = 0 | real-time factor (median) |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-tts-0.6b | default | warm_male | 22 | 0.0069 | 3 | 3 | 0.0152 | 1 | 1.099 |
| qwen3-tts-0.6b | conservative | warm_male | 22 | 0.0083 | 3 | 4 | 0.0216 | 1 | 1.079 |

### By text

| text | focus | engine | phonetic error (mean) | flagged words |
|---|---|---|---|---|
| numbers_dates | numbers, dates and times written out | qwen3-tts-0.6b | 0.006 | el→e[n], el→e[n] |
| money_percent | quantities, currency and percentages | qwen3-tts-0.6b | 0.0029 | — |
| z_c_distinction | Castilian z / soft c (distinción) against s | qwen3-tts-0.6b | 0.0 | — |
| proper_names | Spanish and regional proper names | qwen3-tts-0.6b | 0.0601 | Sanxenxo→san[z]en[∅]so, Sanxenxo→san[z]en[∅]so, Sanxenxo→san[z]en[∅]so, Xavier→[∅]s[ji]abier, Sanxenxo→san[z]en[∅]so |
| jota_and_g | jota and soft g | qwen3-tts-0.6b | 0.0 | — |
| trilled_r | trilled r and tap r | qwen3-tts-0.6b | 0.0 | — |
| ll_y_enye | ll, y and ñ | qwen3-tts-0.6b | 0.0 | — |
| long_words | long, rare words | qwen3-tts-0.6b | 0.0121 | — |
| loanwords_tech | technical vocabulary and loanwords | qwen3-tts-0.6b | 0.0024 | — |
| questions | question and exclamation intonation | qwen3-tts-0.6b | 0.0 | — |
| long_sentence | a long sentence with subordinate clauses | qwen3-tts-0.6b | 0.0 | — |

### Flagged by the phonetic judge, perfect for Whisper

| engine | take | flagged words | Whisper heard |
|---|---|---|---|
| qwen3-tts-0.6b | default_numbers_dates_seed1.wav | el→e[n] | El tren sale el martes 23 de marzo a las 7 y cuarto y llega a Zaragoza a las 10 menos 20. |
| qwen3-tts-0.6b | conservative_numbers_dates_seed1.wav | el→e[n] | El tren sale el martes 23 de marzo a las 7 y cuarto y llega a Zaragoza a las 10 menos 20. |
<!-- results:end -->

## Conclusion (qwen3-tts-0.6b)

- **Castilian Spanish is essentially solved for everyday text.** Eight of the eleven texts have no flagged word in any
  of the four takes: z/c against s, jota, trilled r, ll/y/ñ, loanwords, questions and a 40-word sentence all come out
  clean. Numbers, dates, money and percentages written out in words are read correctly too.
- **Regional proper names are the weak spot.** Almost all of the remaining error is *Sanxenxo* (Galician x, heard as
  "San Censo") and, in some takes, *Xavier* heard as "Javier": both use an x that is not Castilian. The other names
  (*Girona*, *Jerez de la Frontera*, *Joaquín*, *Ainhoa*, *Begoña*) are fine. If your text has names with Galician,
  Catalan or Basque spelling, check them, or spell them the way they should sound.
- **Conservative sampling does not help pronunciation.** It is slightly worse, if anything (phonetic error 0.0083
  against 0.0069; Whisper WER 0.022 against 0.015). The model's default settings are a sound choice for accuracy.
- **Whisper and the phonetic judge mostly agree here.** With a model this accurate, the judge only caught something
  Whisper missed in two takes (the same sentence and seed under both settings): a minor *el → en* in connected speech. The phonetic judge earns its keep on weaker
  models, where Whisper's language model hides real mispronunciations.

## Limitations

- Two seeds per text and setting (44 takes in total); a single rare failure could be missed.
- The phonetic judge does not reliably detect seseo or a lost diphthong in very frequent words (see its docstring),
  so "z/c distinction: 0 errors" means no audible mistake was found, not that distinción is guaranteed.
- Speed is measured with the reference `qwen-tts` implementation on one RTX 4090 (real-time factor ≈ 1.1, no
  streaming or batching). A serving engine such as vLLM-Omni is much faster; speed is not what this experiment
  measures.
