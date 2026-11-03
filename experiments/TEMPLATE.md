# NN · Short title

**Question.** One sentence: what do we want to know?

**Hypothesis.** What we expect and why (optional, but write it before running).

**Setup.** Engines, voice, texts, seeds and the variables that change. Parameters live in `config.yaml`, never
hardcoded in `run.py`. An experiment that compares models lists them under `engines:`, each with its variants
(setting overrides on top of the engine defaults):

```yaml
voice: warm_male
seeds: [0, 1]
engines:
  qwen3-tts-0.6b:
    default: {}
    conservative: {temperature: 0.6, top_k: 20}
```

**How to run.**

```bash
uv run python experiments/NN-short-title/run.py [--engines name,...] [--rescore]
```

Audio goes to `results/NN-short-title/<engine>/` (git-ignored). `run.py` stores one entry per engine in
`results.json` (see `labtts.records.save_engine_results`) and fills the block below from it. `--engines` runs a subset
and keeps the others' results; `--rescore` judges the audio already generated again without regenerating it.

<!-- results:start -->
<!-- results:end -->

## Conclusion (engine)

What the numbers say, in plain words.

## Limitations

What this does not show (sample size, proxies, hardware, listening not done...).
