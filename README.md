# Tokenization Strategy for Japanese LLM Tasks

This repository studies how Japanese input tokenization style affects LLM task performance. I evaluate three tokenization strategies (`baseline`, `character`, `morphology`) across five datasets, five model configurations, and three `length_multiplier` settings. In the current canonical run, `morphology` is roughly accuracy-neutral/slightly positive overall versus `baseline`, while `character` is generally worse.

![detokenization visualization](detokenization-visualization.svg)

## Methodology

### Task families and what they measure

- `multiple_choice` (`JCommonsenseQA`): commonsense selection accuracy.
- `nli` (`JNLI`): premise-hypothesis entailment judgment.
- `extraction` (`JSQuAD`): answer extraction from context.
- `correction` (`JWTD`): typo detection/correction quality.
- `char_counting` (`CharCount`): fine-grained symbol-level counting.

### Strategy definitions

- `baseline`: no forced detokenization.
    - example: `猫が魚を食べた。`
- `character`: split into character units.
    - example: `猫 が 魚 を 食 べ た 。`
- `morphology`: split by morphological tokens via `fugashi` (`-Owakati`).
    - example (typical): `猫 が 魚 を 食べ た 。`

See implementation in `src/tokenizer.py`.

## Key Findings (Canonical Run)

Canonical run file: `data/results/20260215_115324.json`

### Global strategy comparison

| strategy   |         avg score |    delta vs baseline |              cost |
| ---------- | ----------------: | -------------------: | ----------------: |
| baseline   | 62.20% (0.621978) |                    - | $0.628888 (1.00x) |
| character  | 59.79% (0.597903) | -2.41 pp (-0.024075) | $0.907878 (1.44x) |
| morphology | 62.28% (0.622776) | +0.08 pp (+0.000797) | $0.823223 (1.31x) |

### Where effects were strongest

| slice                                             | notable effect            |                                magnitude | implication                                                   |
| ------------------------------------------------- | ------------------------- | ---------------------------------------: | ------------------------------------------------------------- |
| dataset = `JWTD`                                  | `morphology` > `baseline` |                     +1.60 pp (+0.015985) | Tokenization can help harder correction tasks.                |
| dataset = `JCommonsenseQA`                        | `character` < `baseline`  |                     -6.44 pp (-0.064444) | Character splitting hurts high-accuracy MCQ behavior.         |
| model = `google/gemini-2.5-flash-lite:floor:high` | `morphology` > `baseline` |                     +2.61 pp (+0.026135) | Benefit is model-dependent.                                   |
| model = `qwen/qwen3-8b:floor:none`                | `character` < `baseline`  |                     -7.11 pp (-0.071116) | Character strategy regresses strongly on some models.         |
| length multiplier = `10` (vs `1`)                 | lower quality trend       | baseline: -1.90 pp, morphology: -3.39 pp | Bigger context pressure did not improve results in aggregate. |

### Interpretation note

The strategy effects in this run are small overall and should be treated as preliminary.

- Global `morphology` vs `baseline` is near-zero (`+0.08 pp`), while `character` is modestly negative (`-2.41 pp`).
- This low sensitivity appears across heterogeneous model families (Gemini, Qwen, Mistral) with different tokenizer/vocabulary characteristics.
- Despite major model differences in Japanese proficiency, scale, and cost tiers, detokenization strategy usually has a much smaller impact than model choice itself.

## Experiment Snapshot

- `n`: `30` samples per dataset/model/length multiplier
- `seed`: `0`
- `strategies`: `baseline`, `character`, `morphology`
- `length_multipliers`: `1`, `5`, `10`
- datasets:
    - `JCommonsenseQA` (multiple choice)
    - `JNLI` (natural language inference)
    - `JSQuAD` (extraction)
    - `JWTD` (typo correction)
    - `CharCount` (synthetic character counting)
- model configurations:
    - `google/gemini-2.5-flash-lite:floor:none`
    - `google/gemini-3-flash-preview:floor:none`
    - `google/gemini-2.5-flash-lite:floor:high`
    - `qwen/qwen3-8b:floor:none`
    - `mistralai/mistral-small-3.2-24b-instruct:floor:none`

## Reproducibility

### Prerequisites

- Python `>=3.12` (from `pyproject.toml`)
- `uv`
- API/env configuration from `.env.example`

Required env keys:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

Optional env keys:

- `HF_TOKEN` (for higher rate limits)

`OPENAI_API_KEY` + `OPENAI_BASE_URL` support OpenAI-compatible providers (for example OpenRouter and other compatible gateways).

### Setup

```bash
uv sync
cp .env.example .env
# edit .env with your credentials
```

### Run the batch experiment

```bash
uv run python src/run/index.py
```

Result files are written to:

- `data/results/<YYYYMMDD_HHMMSS>.json`

## Current limitations

- Small sample size per cell (`n=30`) can make small deltas unstable.
- Task families differ in difficulty and score distributions, so global averages can hide subgroup effects.

## Citation and Acknowledgements

- Datasets: JGLUE subsets (`JCommonsenseQA`, `JNLI`, `JSQuAD`) and JWTD/CharCount task sources implemented in this repo.
- Tooling: `fugashi` with `unidic-lite` for morphology-level tokenization.
