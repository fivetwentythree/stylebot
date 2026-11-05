# StyleBot White

Prepare a prose-rewriting dataset from E. B. White’s *Writings from the New Yorker* and *Essays of E. B. White*. The dataset follows [Dan MacKinlay’s Stylebot](https://github.com/danmackinlay/stylebot). An ordinary synthetic draft is the input, and White’s source passage is the target.

```text
scripts/
  prepare.py            Split the supplied text into pieces and passages
  merge_essays.py       Add the essay collection and remove overlapping passages
  synthesize_ollama.py Produce synthetic drafts with a local Ollama model
  synthesize.swift       Produce synthetic drafts with Apple’s on-device model
  package.py             Check drafts and export Stylebot JSONL
examples/
  pairs.jsonl            One invented example with the actual record structure
  policy.json            The E. B. White training prompt for Stylebot
```

The book text and generated pairs stay in local files. The public repository contains the scripts and an invented example that demonstrates the format.

## Current dataset

The two collections yielded 963 distinct candidate passages. The first batch contains 167 accepted training pairs and 49 external evaluation pairs. Of the training pairs, 82 come from *Writings from the New Yorker* and 85 come from *Essays of E. B. White*. The other passages are indexed locally for generation or review.

No exact 24-word sequence crossed the training and evaluation split. The anthology’s *Here Is New York* is the only copy in the corpus.

## Prepare and generate

Use Python 3.11 or later. The extractors expect the supplied editions’ text export layouts. Each article or essay stays in one split. The preparation step joins or divides prose into passages, skips known verse or heavily quoted material, and corrects obvious transcription artifacts such as line-wrap hyphenation and small-cap openings. Start the local Ollama server before running the generator.

```sh
python3 scripts/prepare.py "/path/to/new-yorker.txt" work/white
python3 scripts/merge_essays.py "/path/to/essays.txt" work/white work/white-combined
ollama pull qwen3:4b-instruct-2507-q4_K_M
python3 scripts/synthesize_ollama.py work/white-combined/passages.jsonl work/white-combined/qwen-output.jsonl
```

The merge preserves the earlier piece IDs and holdouts. It removes new passages with exact 24-word overlap or substantial overlap in 12-word fingerprints. If you already generated drafts from the first book, carry their JSONL files into the combined working directory to reuse those drafts. The passage IDs still resolve.

With Apple Intelligence available, the Swift script is a second local generator. Its second argument is the number of concurrent requests.

```sh
swift scripts/synthesize.swift work/white-combined/passages.jsonl 2 > work/white-combined/drafts.jsonl
```

## Package and verify

The packager uses the upstream Stylebot validator. It checks length, copying, numeric literals, quotations, capitalized terms, narrator perspective and paragraph breaks. These checks catch some fidelity errors. They leave semantic judgment to spot review and evaluation.

```sh
STYLEBOT='git+https://github.com/danmackinlay/stylebot.git@0101733d68e53221068994444ed3bbb7c4dc92ce'
uv run --with "$STYLEBOT" python scripts/package.py work/white-combined data/eb-white
```

The packaged dataset has six files.

| File | Purpose |
| --- | --- |
| `pairs.jsonl` | Accepted training pairs in Stylebot format |
| `evaluation/pairs.jsonl` | Accepted pairs from whole pieces held out before generation |
| `policy.json` | The matching E. B. White system prompt |
| `pieces.jsonl` | Piece titles, dates, source lines and split assignments |
| `passages.jsonl` | Candidate targets and pairing status |
| `dataset-summary.json` | Counts, coverage and quality-check outcomes |

## Train Qwen3.5-9B

Set `TINKER_API_KEY` in your environment, then run Qwen3.5-9B training on Tinker from the repository root.

```sh
DATA="$PWD/data/eb-white"
uvx --from "stylebot[trainer] @ $STYLEBOT" ai-style train \
  --data-dir "$DATA" --policy "$DATA/policy.json" \
  --work-dir "$PWD/work/train" --manifest-out "$PWD/work/run.json" \
  --base-model Qwen/Qwen3.5-9B --renderer qwen3_5_disable_thinking \
  --price-per-mtok 1.463
```

The trainer estimates the cost and asks for confirmation before launching the Tinker job. It creates an internal validation split by whole piece from `pairs.jsonl`. The separate evaluation file stays outside the training input.
