# 02 · Phonetics in Castilian Spanish

**Question.** How accurately does each engine pronounce Castilian Spanish, and do its sampling settings matter?

**Setup.** Eleven short texts ([`texts/phonetics.jsonl`](../../texts/phonetics.jsonl)), each aimed at a known
difficulty: numbers, z/c versus s, proper names, trilled r, jota, ll/y/ñ, long words, loanwords, questions and a long
sentence. Engines that clone use the designed `warm_male` voice. Two seeds per text, engine and variant
([`config.yaml`](config.yaml)); for `qwen3-tts-0.6b`, `default` is the model's own settings (temperature 0.9, top-k
50) and `conservative` lowers them to 0.6 and 20; for `ming-tts-0.5b`, `default` is its API's own settings (see
[`labtts/engines/ming_tts.py`](../../labtts/engines/ming_tts.py)). Every take is judged twice: by Whisper (WER) and by the
[phonetic judge](../../labtts/judges/phonetic.py).

**How to run.** `uv run python experiments/02-phonetics/run.py [--engines name,...] [--rescore]` (needs the voices
from experiment 01).

<!-- results:start -->
### By engine and variant

| engine | variant | voice | takes | phonetic error (mean) | takes with flagged words | flagged words | Whisper WER (mean) | flagged but WER = 0 | real-time factor (median) |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-tts-0.6b | default | qwen3-tts-1.7b-voicedesign/warm_male | 22 | 0.0069 | 3 | 3 | 0.0152 | 1 | 1.099 |
| qwen3-tts-0.6b | conservative | qwen3-tts-1.7b-voicedesign/warm_male | 22 | 0.0083 | 3 | 4 | 0.0216 | 1 | 1.079 |
| audio8-air | cloned | qwen3-tts-1.7b-voicedesign/warm_male | 22 | 0.0099 | 4 | 5 | 0.0343 | 0 | 1.095 |
| audio8-air | built_in_voice | built-in | 22 | 0.0145 | 5 | 7 | 0.0357 | 1 | 1.063 |
| ming-tts-0.5b | default | qwen3-tts-1.7b-voicedesign/warm_male | 22 | 0.0089 | 3 | 5 | 0.0091 | 2 | 0.969 |

### By text

| text | focus | engine | phonetic error (mean) | flagged words |
|---|---|---|---|---|
| numbers_dates | numbers, dates and times written out | qwen3-tts-0.6b | 0.006 | el→e[n], el→e[n] |
| numbers_dates | numbers, dates and times written out | audio8-air | 0.0089 | — |
| numbers_dates | numbers, dates and times written out | ming-tts-0.5b | 0.006 | a→[∅] |
| money_percent | quantities, currency and percentages | qwen3-tts-0.6b | 0.0029 | — |
| money_percent | quantities, currency and percentages | audio8-air | 0.0029 | — |
| money_percent | quantities, currency and percentages | ming-tts-0.5b | 0.0057 | — |
| z_c_distinction | Castilian z / soft c (distinción) against s | qwen3-tts-0.6b | 0.0 | — |
| z_c_distinction | Castilian z / soft c (distinción) against s | audio8-air | 0.0125 | zanahoria→zana[u]ri[l]a |
| z_c_distinction | Castilian z / soft c (distinción) against s | ming-tts-0.5b | 0.0063 | — |
| proper_names | Spanish and regional proper names | qwen3-tts-0.6b | 0.0601 | Sanxenxo→san[z]en[∅]so, Sanxenxo→san[z]en[∅]so, Sanxenxo→san[z]en[∅]so, Xavier→[∅]s[ji]abier, Sanxenxo→san[z]en[∅]so |
| proper_names | Spanish and regional proper names | audio8-air | 0.0664 | Sanxenxo→san[z]en[∅]so, Sanxenxo→san[∅]sen[∅]so, Ainhoa,→a[ñ]oa, Sanxenxo→san[∅]sen[∅]so, Ainhoa,→a[ñ]oa, Ainhoa,→a[ñ]oa |
| proper_names | Spanish and regional proper names | ming-tts-0.5b | 0.0633 | Sanxenxo→san[∅]sen[∅]so, Ainhoa,→ainoa[ai], Xavier→ks[e]bie[∅] |
| jota_and_g | jota and soft g | qwen3-tts-0.6b | 0.0 | — |
| jota_and_g | jota and soft g | audio8-air | 0.0 | — |
| jota_and_g | jota and soft g | ming-tts-0.5b | 0.0 | — |
| trilled_r | trilled r and tap r | qwen3-tts-0.6b | 0.0 | — |
| trilled_r | trilled r and tap r | audio8-air | 0.0 | — |
| trilled_r | trilled r and tap r | ming-tts-0.5b | 0.0 | — |
| ll_y_enye | ll, y and ñ | qwen3-tts-0.6b | 0.0 | — |
| ll_y_enye | ll, y and ñ | audio8-air | 0.0 | — |
| ll_y_enye | ll, y and ñ | ming-tts-0.5b | 0.0075 | a→[∅] |
| long_words | long, rare words | qwen3-tts-0.6b | 0.0121 | — |
| long_words | long, rare words | audio8-air | 0.0437 | electroencefalograma→elektroen[s]falograma, electroencefalograma→elektroen[∅]falograma, electroencefalograma→elektroen[∅]falograma, Desoxirribonucleico,→d[i]soksi[r]ibonukl[∅]iko, electroencefalograma→elektroen[∅]falogra[m]ma |
| long_words | long, rare words | ming-tts-0.5b | 0.0097 | — |
| loanwords_tech | technical vocabulary and loanwords | qwen3-tts-0.6b | 0.0024 | — |
| loanwords_tech | technical vocabulary and loanwords | audio8-air | 0.0 | — |
| loanwords_tech | technical vocabulary and loanwords | ming-tts-0.5b | 0.0 | — |
| questions | question and exclamation intonation | qwen3-tts-0.6b | 0.0 | — |
| questions | question and exclamation intonation | audio8-air | 0.0 | — |
| questions | question and exclamation intonation | ming-tts-0.5b | 0.0 | — |
| long_sentence | a long sentence with subordinate clauses | qwen3-tts-0.6b | 0.0 | — |
| long_sentence | a long sentence with subordinate clauses | audio8-air | 0.0 | — |
| long_sentence | a long sentence with subordinate clauses | ming-tts-0.5b | 0.0 | — |

### Flagged by the phonetic judge, perfect for Whisper

| engine | take | flagged words | Whisper heard |
|---|---|---|---|
| qwen3-tts-0.6b | default_numbers_dates_seed1.wav | el→e[n] | El tren sale el martes 23 de marzo a las 7 y cuarto y llega a Zaragoza a las 10 menos 20. |
| qwen3-tts-0.6b | conservative_numbers_dates_seed1.wav | el→e[n] | El tren sale el martes 23 de marzo a las 7 y cuarto y llega a Zaragoza a las 10 menos 20. |
| audio8-air | built_in_voice_z_c_distinction_seed0.wav | zanahoria→zana[u]ri[l]a | Cecilia cocinó una cazuela de calabacín con cebolla y zanahoria para cenar con sus cinco sobrinos. |
| ming-tts-0.5b | default_numbers_dates_seed0.wav | a→[∅] | El tren sale el martes 23 de marzo a las 7 y cuarto y llega a Zaragoza a las 10 menos 20. |
| ming-tts-0.5b | default_ll_y_enye_seed0.wav | a→[∅] | La lluvia de otoño llenó de charcos el camino que lleva a la montaña donde vive mi abuela. |
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
  Whisper missed in two takes (the same sentence and seed under both settings): a minor *el → en* in connected
  speech. The phonetic judge earns its keep on weaker models, where Whisper's language model hides real
  mispronunciations.

## Conclusion (audio8-air)

- **Good, but a step behind Qwen3-TTS 0.6B.** Cloning the same voice, its phonetic error is 0.0099 against 0.0069
  and its Whisper WER 0.034 against 0.015. With its built-in voice it is a little worse again (0.0145).
- **It drops syllables in long words.** *Electroencefalograma* comes out as "electroenfalograma" in all four takes,
  with either voice, and *desoxirribonucleico* lost a vowel once. Qwen read both correctly every time.
- **Unusual names suffer more.** On top of the *Sanxenxo* and *Xavier* trouble both models share, *Ainhoa* came out
  as "Añoa" in three of the four takes. In both cloned takes, Whisper heard *jamás* as "Hamás".
- **The everyday texts are fine.** Numbers, z/c, trilled r, ll/y/ñ, questions and the long sentence are clean or
  nearly so with both voices.

## Conclusion (ming-tts-0.5b)

- **Close to Qwen3-TTS 0.6B, ahead of Audio8.** Cloning the same voice, its phonetic error is 0.0089 (Qwen 0.0069,
  Audio8 0.0099), and it has the lowest Whisper WER of all (0.009 against Qwen's 0.015): it skips or swaps fewer
  words.
- **Its errors are the shared ones.** Almost all of them are the non-Castilian names: *Sanxenxo* ("San Senso"),
  *Xavier* ("Ksebie") and, once, *Ainhoa* with an extra vowel. The only other flags are a dropped *a* in connected
  speech. Unlike Audio8, it reads the long words (*electroencefalograma*, *desoxirribonucleico*) correctly.

## Limitations

- Two seeds per text and variant (22 takes each); a single rare failure could be missed.
- The phonetic judge does not reliably detect seseo or a lost diphthong in very frequent words (see its docstring),
  so "z/c distinction: 0 errors" means no audible mistake was found, not that distinción is guaranteed.
- Speed is measured with each model's reference implementation on one RTX 4090 (real-time factor ≈ 1.0–1.1 for all
  three, no streaming or batching). Serving engines such as vLLM-Omni or SGLang-Omni are much faster; speed is not what this
  experiment measures.
