# Project folders

The Tinker tutorial configures an output `log_path`; it does not prescribe this repository's folder names. This layout keeps source preparation, frozen experiments and training results separate while retaining the Cookbook's standard output files. [SFT with Config](https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/sft-with-config/)

```text
StyleBot/
  AGENTS.md                       Python commands must use uv
  pyproject.toml / uv.lock         Pinned dependencies
  .python-version                 Python 3.11
  .env.example                    Empty credential template
  .env                            Your future local key; ignored
  configs/qwen3.5-9b-v1.json       First-experiment settings and price assumptions
  prompts/baseline-style.txt       Explicit style prompt for the strong baseline
  examples/
    pairs.jsonl                   Invented example of a training pair
    policy.json                   Author-free rewrite instruction
    evaluation-inputs.jsonl        Twenty invented evaluation cases
  scripts/                        Extraction, review, preparation, training and comparison
  tests/                          Offline regression checks
  docs/                           Run guide, this map and review rubric

  Writings/                       Private source books, when stored here
  work/
    white-combined/               Piece boundaries, passages, drafts and exact-pair reviews
    experiment-v1/                Additional held-out cases and preparation logs
    huggingface/hub/              Tokenizer and model configuration; no model weights
  data/
    eb-white/                     Reviewed master corpus and frozen original evaluation
    tinker/white-v1/
      train.jsonl                 162 system/user/assistant conversations
      validation.jsonl            18 conversations, separated by whole piece
      evaluation.jsonl            50 test cases: 30 reference pairs and 20 invented inputs
      policy.json                 Frozen training and inference instruction
      baseline-style.txt          Frozen strong-baseline instruction
      config.json                 Frozen experiment settings
      splits.json                 Passage IDs, sources and target fingerprints
      manifest.json               File fingerprints, source identity and counts
      token-report.json           Actual token counts, loss checks, versions and cost estimate
      tokenized/
        train.jsonl               Serialized Tinker Datum objects
        validation.jsonl          Serialized Tinker Datum objects
  runs/
    README.md                     Public explanation of runtime output
    white-v1/
      baselines/                  Outputs captured before training
      tinker/                     Cookbook log_path
        config.json               Created by real training
        metrics.jsonl             Created by real training
        checkpoints.jsonl         Created by real training
        logs.log                  Created by real training
      outputs/                    Tuned-model evaluation rewrites
      review/                     Blinded packet, ratings and separate answer key
      run.json                    Actual checkpoint and exact inference instruction
```

`Writings/`, `work/`, `data/`, `.env`, `.venv/` and runtime files under `runs/` are ignored. The source books and their derived passages are not published to GitHub. Runtime subdirectories are prepared locally; metrics, outputs and checkpoint records do not exist until their corresponding real jobs run. Extra logs may be produced by the Cookbook.

The frozen experiment retains the master dataset's fingerprints. Changing examples, prompts or settings requires a new experiment directory, such as `white-v2`, rather than editing the first experiment in place.

## The evaluation input file

`examples/evaluation-inputs.jsonl` contains one JSON object per line. Each object has:

| Field | Meaning |
| --- | --- |
| `id` | Stable identifier for matching outputs across models |
| `kind` | `novel`, distinguishing it from a White reference pair |
| `source` | Invented scenario identifier, not a book citation |
| `title` | Short label for review |
| `input` | Ordinary prose to rewrite |
| `fact_checks` | Facts and qualifications a faithful rewrite must preserve |
| `focus` | Behaviors being tested, such as chronology or uncertainty |

These 20 inputs are copied into the private, frozen `evaluation.jsonl`. They never become training examples. There is no target rewrite for them: their purpose is to test stylistic transfer while checking factual fidelity. Modern subjects help reveal forced historical props or memorized author associations. One case contains a quoted instruction, which must be treated as source text.

The other 30 evaluation cases include an ordinary draft and a genuine held-out White reference. Neither the reference nor the fact checklist is sent to the model during sampling. Reviewers judge generated prose using the input, checklists and review rubric.
