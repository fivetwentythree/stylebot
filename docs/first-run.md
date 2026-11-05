# First Qwen3.5-9B experiment

The local dataset and token preparation are complete. No Tinker key was needed for this work. Paid baseline sampling and training have not started.

This project uses the recommended Cookbook dataset-builder and configuration workflow, followed by its standard trainer. It uses the matching non-thinking Qwen renderer and trains only on the final assistant response. [SFT with Config](https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/sft-with-config/) · [Rendering](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/)

## Already prepared

| Item | First experiment |
| --- | --- |
| Model | Qwen/Qwen3.5-9B |
| Reviewed master pool | 180 pairs from 105 pieces |
| Training / validation | 162 / 18, separated by whole piece |
| Evaluation | 30 White reference cases + 20 invented cases |
| LoRA rank / batch size | 32 / 8 |
| Epochs / updates | 3 / 63 |
| Learning rate | Cookbook heuristic: 0.000472979, linear decay |
| Maximum length | 4,096; longest prepared sequence 773 tokens |
| Prompt | Saved author-free rewrite instruction |
| Training loss | Final assistant response only |

The canonical training pool and all 49 original evaluation records remain unchanged. Only the 10 approved original evaluation pairs enter scoring, alongside 20 additional held-out pairs and the invented cases. Evaluation does not enter the training or validation files.

All model computation takes place on Tinker. The Mac runs preparation and the client. Only tokenizer/configuration files have been fetched, not Qwen weights. Ollama is unnecessary.

## Environment

Run all commands from the repository root. Always use `uv` for Python.

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
```

For a new experiment, the preparation commands are:

```sh
uv run python scripts/prepare_experiment.py --output data/tinker/white-v2
uv run python scripts/validate_tinker.py --prepared data/tinker/white-v2
```

The first experiment is already present under `data/tinker/white-v1`. Preparation refuses to overwrite a nonempty directory. Validation checks actual renderer tokens, assistant-only loss weights, serialization and truncation without creating a Tinker client. It may fetch tokenizer/configuration files. Once a report exists, its tokenizer revision and SDK versions are retained. A fresh GitHub checkout does not contain the private corpus or held-out cases; it cannot reconstruct them from the public examples alone.

## When the key is available

Create a local `.env` file using `.env.example` as the template and fill in `TINKER_API_KEY` there. The file is ignored by Git. Do not put the key into a committed file or a chat message. `uv run --env-file .env` loads it only for the requested process.

Before spending, check the prices in the frozen configuration against [Tinker's current pricing](https://thinkingmachines.ai/tinker/pricing/). If pricing changes, prepare a new experiment with updated assumptions. The current report estimates $0.4652 for all training and scheduled validation; each 50-case comparison arm has an output-limit estimate below $0.035. Total estimated token charges are about $0.57, excluding storage. The allowances below permit those estimates; they are pre-launch checks, not billing caps enforced by Tinker.

First save both baseline arms, using the same frozen evaluation inputs and sampling settings:

```sh
uv run --env-file .env python scripts/compare_models.py sample \
  --arm base-plain --max-cost-usd 0.05
uv run --env-file .env python scripts/compare_models.py sample \
  --arm base-style --max-cost-usd 0.05
```

`base-plain` uses the author-free project prompt. `base-style` uses the explicit White-style prompt. Both sample the untrained base model. These are comparison samples, not two training runs. Each saves completed cases immediately; rerunning resumes missing cases and rejects stale or mismatched results.

Launch the actual training run once the two baseline files are complete:

```sh
uv run --env-file .env python scripts/train_tinker.py --max-cost-usd 0.60
```

The launcher verifies frozen data and baseline identities before opening the training client. The standard Cookbook loop evaluates validation loss every ten steps, saves checkpoints every twenty and saves a final checkpoint. It preserves the last two-example batch, so every training example is used each epoch. Interrupted runs resume from the most recent saved optimizer checkpoint in the same run directory; work after that checkpoint may need repeating and can incur additional charges. Repeating a completed launcher invocation leaves the saved completed run in place.

## Compare the result

```sh
uv run --env-file .env python scripts/compare_models.py sample \
  --arm tuned --max-cost-usd 0.05
uv run python scripts/compare_models.py packet
```

Read `runs/white-v1/review/comparison.md` alongside [the evaluation rubric](evaluation-rubric.md). Fill `ratings.jsonl` with each candidate's style score, fidelity decision and notes, then record a preference or tie. Keep `blind-key.json` closed until the ratings are complete. It maps the shuffled A/B/C candidates back to the three arms. Packet generation refuses to overwrite existing ratings.

The 20 invented cases test transfer to unfamiliar material. The 30 reference cases test rewriting toward the held-out voice. Neither is proof of exact imitation, and these cases must remain reserved for evaluation. Repeatedly choosing hyperparameters against this suite would make it a development set; reserve a new untouched set before making stronger claims.

Inspect the validation curve as well as the writing. The final checkpoint is a starting candidate, not automatically the best checkpoint. A later experiment can compare learning rates, fewer epochs or more reviewed data using the same controlled process.

## Use the trained rewriter

After a checkpoint exists:

```sh
uv run --env-file .env python scripts/rewrite.py draft.md \
  --manifest runs/white-v1/run.json \
  --policy data/tinker/white-v1/policy.json > rewritten.md
```

The saved manifest supplies the trained checkpoint, model, renderer and exact author-free instruction. No special style tag is required. This uses Tinker's sampler; local adapter export and local model hosting are separate future work.
