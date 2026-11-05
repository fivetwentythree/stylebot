# StyleBot White

StyleBot White is an independent project by Lochana Perera ([fivetwentythree](https://github.com/fivetwentythree)) to build an E. B. White prose rewriter. It draws on *Writings from the New Yorker* and *Essays of E. B. White*. The project’s corpus selection, evaluation design, and development direction are Lochana’s. An ordinary synthetic draft is the input, and White’s source passage is the target.

Dan MacKinlay’s writing about reverse-edit fine-tuning was an inspiration for the project. [Fine-tuning danbot](https://danmackinlay.name/notebook/fine_tuning_danbot.html)

```text
scripts/
  prepare.py           Split the supplied text into pieces and passages
  merge_essays.py      Add the essay collection and remove overlapping passages
  review_batch.py      Select training passages and record exact-pair reviews
  audit_pairs.py       Validate a complete review of an older accepted snapshot
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

The two collections contain 963 indexed candidate passages. The reviewed training pool has 180 pairs from 105 pieces. Of these, 93 come from *Writings from the New Yorker* and 87 come from *Essays of E. B. White*. Other passages remain indexed locally for generation or review.

All 216 pairs in the older accepted pool received a semantic review. The original drafts often passed mechanical checks while losing details or reversing meanings. The resulting decisions are recorded against the exact target, draft, generator, source and original record identity.

| Older pool | Kept unchanged | Repaired | Set aside |
| --- | --- | --- | --- |
| 167 training pairs | 30 | 126 | 11 |
| 49 evaluation pairs | 10 eligible for scoring | 0 | 39 excluded from scoring |

The 180 training pairs comprise the 30 retained older drafts, 126 corrected older drafts and 24 additions from the preceding batch. All have recorded semantic approvals. Repairs retain their original draft fingerprints and generator attribution in `meta.parent_pair`. These are Codex reviews, including self-review of Codex corrections, rather than independent human certification. No paid training has started.

All 49 original evaluation records remain byte-for-byte unchanged. A separate scoring file contains the 10 that passed review. Eligibility was decided before any trained-model outputs existed. This is a small scoring set and needs broader coverage before drawing strong conclusions about style quality. An earlier on-device generation batch screened 84 drafts, rejected all 84, and supplied the targets for the 24 subsequently corrected additions.

No exact 24-word sequence crossed the training and evaluation split. The anthology’s *Here Is New York* is the only copy in the corpus.

## Voice instruction

The system prompt begins “Rewrite the supplied prose in your trained editorial voice.” It does not name White. Every pair uses the same instruction, with White’s prose as the assistant target. The name and source attribution remain in project documentation and provenance.

Once trained, select the fine-tuned checkpoint or load its adapter to use the voice. The application supplies the saved system instruction automatically. No special tag is required for this single-voice model. Tags would be a separate design choice if one model were later trained to switch among several voices.

Dataset metadata tags such as `eb-white` are bookkeeping labels. They are not included in the chat messages and do not need to be typed by the user.

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

Select unfinished training passages with neighboring context. Evaluation pieces, accepted passages and targets in `work/white-combined/target-exclusions.json` are excluded. Target exclusions cover an extensively attributed translation, an uncertain transcription and an abbreviated observation ledger. Context is for interpreting references and stops outside the target span. It does not enter the training messages.

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

## Review an older pool

Save the accepted training and evaluation records in `work/white-combined/legacy-audit/pairs-before.jsonl` before editing anything. Review every pair and record one decision per passage in `notes.jsonl`. Each decision includes `id`, `split`, `source`, `generator`, `capture_id`, `target_sha256`, `draft_sha256`, `verdict`, `reviewer` and a specific `note`.

```sh
python3 scripts/audit_pairs.py \
  work/white-combined/legacy-audit/pairs-before.jsonl \
  work/white-combined/legacy-audit/notes.jsonl \
  work/white-combined/legacy-audit/validated.jsonl
```

The validator requires complete snapshot coverage and matching content and provenance. Once the audit exists, the packager admits historical training drafts only when their exact version has passed. A different older generator output cannot silently replace a rejected draft. Corrected training drafts pass through `review_batch.py record` into `legacy-repaired-reviewed-output.jsonl`, with fresh reviews and fingerprints. Evaluation drafts are not repaired. The original evaluation file, audit decisions and approved scoring subset are separate exports.

## Package and verify

The packager uses the upstream Stylebot validator. It checks length, copying, numeric literals, quotations, capitalized terms, narrator perspective and paragraph breaks. Reviewed-batch files require a passing fidelity decision with matching hashes. Historical generator files remain readable and are subject to the legacy audit when present. Mechanical checks cannot establish semantic fidelity by themselves. For example, the review found that a Roman numeral in “Volume I” had concealed a missing first-person narrator; that reference syntax is now excluded from the pronoun check.

```sh
STYLEBOT='git+https://github.com/danmackinlay/stylebot.git@0101733d68e53221068994444ed3bbb7c4dc92ce'
uv run --with "$STYLEBOT" python scripts/package.py work/white-combined data/eb-white
```

The packaged dataset contains these exports.

| File | Purpose |
| --- | --- |
| `pairs.jsonl` | Accepted training pairs in Stylebot format |
| `evaluation/pairs.jsonl` | Frozen original evaluation records from whole held-out pieces |
| `evaluation/audit.jsonl` | Exact-pair review decisions and scoring eligibility |
| `evaluation/approved-pairs.jsonl` | Unmodified records that passed review, for scoring |
| `training-inputs/accepted.jsonl` | Training-only records for the policy-backed training view |
| `policy.json` | The matching voice prompt and evaluation-source exclusion list |
| `pieces.jsonl` | Piece titles, dates, source lines and split assignments |
| `passages.jsonl` | Candidate targets and pairing status |
| `dataset-summary.json` | Counts, coverage and quality-check outcomes |

The packager defaults to `examples/policy.json` and accepts `--policy` for an explicit replacement. The author-free prompt migration preceded the older-pair audit. During this audit, the evaluation messages and metadata remained unchanged. Changing the prompt later would produce a new evaluation version and require a new snapshot.

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
