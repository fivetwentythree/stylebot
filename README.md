# StyleBot White

StyleBot White is an independent project by Lochana Perera ([fivetwentythree](https://github.com/fivetwentythree)) to build an E. B. White prose rewriter. It draws on *Writings from the New Yorker* and *Essays of E. B. White*. The project’s corpus selection, evaluation design, and development direction are Lochana’s. An ordinary synthetic draft is the input, and White’s source passage is the target.

Dan MacKinlay’s writing about reverse-edit fine-tuning was an inspiration for the project. [Fine-tuning danbot](https://danmackinlay.name/notebook/fine_tuning_danbot.html)

```text
scripts/
  prepare.py           Split the supplied text into pieces and passages
  merge_essays.py      Add the essay collection and remove overlapping passages
  review_batch.py      Select training passages and record exact-pair reviews
  synthesize.swift     Optional drafts from Apple’s on-device model
  synthesize_ollama.py Optional drafts from an existing local Ollama model
  package.py           Check drafts and export training and evaluation files
  rewrite.py           Use a trained checkpoint with its matching voice prompt
examples/
  pairs.jsonl          One invented example with the actual record structure
  policy.json          The training and inference prompt, without an author name
tests/
  test_review_pipeline.py
```

The book text and generated pairs stay in local files. The public repository contains the scripts and an invented example that demonstrates the format.

## Current dataset

The two collections contain 963 distinct candidate passages. The current pool has 191 accepted training pairs from 111 pieces and 49 external evaluation pairs from 24 held-out pieces. Of the training pairs, 99 come from *Writings from the New Yorker* and 92 come from *Essays of E. B. White*. Other passages remain indexed locally for generation or review.

The latest batch screened 84 contextual Apple drafts. Seventy failed mechanical checks and the other fourteen failed semantic review, including cases that imported neighboring text. Twenty-four passages were then rewritten directly in Codex and checked again. Those 24 additions have recorded fidelity decisions bound to their exact input and target hashes. This was Codex self-review after repair. The earlier pool has mechanical checks and 15 spot-review decisions. The entire pool still needs broader semantic review before a quality-focused training run. No paid training has started.

No exact 24-word sequence crossed the training and evaluation split. The anthology’s *Here Is New York* is the only copy in the corpus.

## Voice instruction

The system prompt begins “Rewrite the supplied prose in your trained editorial voice.” It does not name White. Every pair uses the same instruction, with White’s prose as the assistant target. The name and source attribution remain in project documentation and provenance.

The hypothesis is that fine-tuning can make this voice the model’s default rewriting behavior. Removing the author’s name is not evidence of better style or guaranteed imitation. That needs blind comparisons on held-out passages and unfamiliar topics. “Use the given style” is ambiguous when no reference style has been supplied. A consistent instruction tied to the trained voice is clearer.

These pairs teach rewriting an existing passage. For writing from a topic, first produce a factual draft and then apply the rewriter. Direct topic-to-essay generation would need its own matching training examples and evaluation.

## Prepare the corpus

Python 3.11 was used for the initial workflow. Python 3.11 or later is required. The extractors expect the supplied editions’ text export layouts. Each article or essay stays in one split. Preparation joins or divides prose into passages, skips known verse or heavily quoted material, and corrects obvious transcription artifacts such as line-wrap hyphenation and small-cap openings.

```sh
python3 scripts/prepare.py "/path/to/new-yorker.txt" work/white
python3 scripts/merge_essays.py "/path/to/essays.txt" work/white work/white-combined
```

The merge preserves the earlier piece IDs and holdouts. It removes new passages with exact 24-word overlap or substantial overlap in 12-word fingerprints. If you already generated drafts from the first book, carry their JSONL files into the combined working directory to reuse those drafts. The passage IDs still resolve.

## Generate and review a batch

Select unfinished training passages with neighboring context. Evaluation pieces and already accepted passages are excluded. Context is for interpreting references and stops outside the target span. It does not enter the training messages.

```sh
python3 scripts/review_batch.py prepare work/white-combined data/eb-white \
  work/white-combined/context-input.jsonl --limit 50
```

Codex can read this batch and write ordinary drafts to `context-drafts.jsonl`. Each output needs `id`, `text`, `model`, `provider`, `strategy`, `prompt_version`, and optionally `temperature`. Keep the source target unchanged. With Apple Intelligence available, the optional Swift generator produces drafts locally. Its second argument controls concurrency. Its current contextual prompt is version 4. The initial contextual batch used version 3.

```sh
swift scripts/synthesize.swift work/white-combined/context-input.jsonl 2 \
  > work/white-combined/context-drafts.jsonl
```

The Ollama generator is another option when a suitable model is already installed. Ollama is unnecessary for Codex generation, Apple generation or Tinker training. The local draft generator and the Qwen3.5-9B training model serve different purposes.

Review each draft against its target. Check every fact, actor, sequence, causal claim, narrator, qualification and quotation. Reject added context, invented details, summaries and excessive copying. A phrase can change wording while still changing its meaning, even when all numbers and names survive. Repairs need a new review decision.

Write decisions to `context-notes.jsonl`. Each decision contains `id`, `verdict` (`pass` or `reject`), `reviewer`, a specific `note`, `target_sha256`, and `draft_sha256`. Hash the exact target and the stripped draft with SHA-256. The recorder refuses stale decisions, duplicate IDs and evaluation targets. A rejected empty generation can be recorded, but cannot be approved.

```sh
python3 scripts/review_batch.py record \
  work/white-combined/context-input.jsonl \
  work/white-combined/context-drafts.jsonl \
  work/white-combined/context-notes.jsonl \
  work/white-combined/context-reviewed-output.jsonl
```

For repairs, use separate `context-repairs.jsonl` and `context-repair-notes.jsonl` files, then run the same recorder to produce `reviewed-output.jsonl`. These names preserve the original batch decisions. Archive completed batches before reusing the working filenames.

## Package and verify

The packager uses the upstream Stylebot validator. It checks length, copying, numeric literals, quotations, capitalized terms, narrator perspective and paragraph breaks. The new reviewed-batch files also require a passing fidelity decision with matching hashes. Historical generator files remain readable to reproduce the earlier pool. Mechanical checks cannot establish semantic fidelity by themselves.

```sh
STYLEBOT='git+https://github.com/danmackinlay/stylebot.git@0101733d68e53221068994444ed3bbb7c4dc92ce'
uv run --with "$STYLEBOT" python scripts/package.py work/white-combined data/eb-white
```

The packaged dataset contains these exports.

| File | Purpose |
| --- | --- |
| `pairs.jsonl` | Accepted training pairs in Stylebot format |
| `evaluation/pairs.jsonl` | Accepted pairs from whole pieces held out before generation |
| `training-inputs/accepted.jsonl` | Training-only records for the policy-backed training view |
| `policy.json` | The matching voice prompt and evaluation-source exclusion list |
| `pieces.jsonl` | Piece titles, dates, source lines and split assignments |
| `passages.jsonl` | Candidate targets and pairing status |
| `dataset-summary.json` | Counts, coverage and quality-check outcomes |

The packager defaults to `examples/policy.json` and accepts `--policy` for an explicit replacement. Changing the prompt re-exports the message and prompt hash for every pair. The latest export changed the evaluation system instruction while preserving all 49 evaluation inputs, targets, source assignments and record IDs.

Regression checks use invented text and make no model calls.

```sh
uv run --with "$STYLEBOT" python -m unittest discover -s tests -v
```

## Train Qwen3.5-9B

Stylebot calls the Tinker Cookbook supervised fine-tuning pipeline. It formats each pair as a chat and trains on the assistant turn containing White’s prose. The renderer disables Qwen3.5 thinking for the rewrite task. [Tinker tutorial 303](https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/sft-with-config/)

Set `TINKER_API_KEY` in your environment, then run Qwen3.5-9B training on Tinker from the repository root.

```sh
DATA="$PWD/data/eb-white"
uvx --from "stylebot[trainer] @ $STYLEBOT" ai-style train \
  --data-dir "$DATA" --draws "$DATA/training-inputs" \
  --policy "$DATA/policy.json" --views-root "$PWD/work/training-views" \
  --work-dir "$PWD/work/train" --manifest-out "$PWD/work/run.json" \
  --base-model Qwen/Qwen3.5-9B --renderer qwen3_5_disable_thinking \
  --price-per-mtok 1.463
```

The trainer estimates the cost and asks for confirmation before launching the Tinker job. It creates an internal validation split by whole piece from the training view. The separate evaluation file stays outside the training input. The policy-backed view also saves this project’s exact system prompt in the run manifest so inference can use the same instruction. The listed price is the supplied rate and should be checked against current Tinker pricing when launching.

After training, the run manifest records the Tinker sampler checkpoint, and Stylebot exports the PEFT adapter to `work/train/peft_adapter` by default. [Tinker checkpoint guide](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/) · [Adapter export](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/lora-adapter/)

## Rewrite after training

The rewrite script reads the trained checkpoint, model, renderer and chunk size from the run manifest. It checks that the supplied policy matches the saved training prompt and refuses a manifest without a trained checkpoint. Use the same renderer for training and sampling. [Tinker rendering guide](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/)

```sh
uv run --with "stylebot[trainer] @ $STYLEBOT" python scripts/rewrite.py draft.md \
  --manifest work/run.json --policy data/eb-white/policy.json > rewritten.md
```

This uses Tinker’s sampler. It does not download the base model to the Mac. The author-free prompt is supplied automatically. Its eventual quality must be measured after training, including content fidelity, style, copying and behavior on unfamiliar subjects.
