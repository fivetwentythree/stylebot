"""Freeze reviewed data into Tinker Cookbook files without contacting Tinker."""

import argparse
import json
import math
from pathlib import Path

from experiment import file_sha, load_config, write_json
from package import check
from review_batch import read_jsonl, text_sha, write_jsonl
from stylebot.pairs import validate_pairs_file
from stylebot.train import assemble_training_corpus


def phrase_windows(text, size=24):
    words = text.casefold().split()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def build_cases(dataset, work, additions, novel):
    training = read_jsonl(dataset / "pairs.jsonl")
    train_sources = {r["meta"]["source"] for r in training}
    originals = read_jsonl(dataset / "evaluation" / "pairs.jsonl")
    original_ids = {r["meta"]["passage_id"] for r in originals}
    passages = {p["id"]: p for p in read_jsonl(work / "passages.jsonl")}
    pieces = {p["source"]: p for p in read_jsonl(work / "pieces.jsonl")}
    cases = []
    for row in read_jsonl(dataset / "evaluation" / "approved-pairs.jsonl"):
        if row not in originals:
            raise ValueError("Existing scoring pair differs from the frozen original")
        meta = row["meta"]
        cases.append(
            {
                "id": "original-" + meta["passage_id"],
                "kind": "reference",
                "passage_id": meta["passage_id"],
                "source": meta["source"],
                "title": meta["source_title"],
                "input": row["messages"][1]["content"],
                "reference": row["messages"][2]["content"],
                "review": {
                    "verdict": "pass",
                    "reviewer": "Codex semantic review of original pair",
                    "input_sha256": text_sha(row["messages"][1]["content"]),
                    "reference_sha256": meta["target_sha256"],
                },
            }
        )
    new_cases = read_jsonl(additions)
    for row in new_cases:
        if row["passage_id"] in original_ids:
            raise ValueError(
                "A new case must not replace a frozen original evaluation case"
            )
    cases.extend(new_cases)
    for row in read_jsonl(novel):
        if row["kind"] != "novel" or not row.get("fact_checks") or row.get("reference"):
            raise ValueError(
                "Novel cases need explicit facts and no invented White reference"
            )
        cases.append(row)
    ids = set()
    train_phrases = set().union(
        *(phrase_windows(r["messages"][i]["content"]) for r in training for i in (1, 2))
    )
    for row in cases:
        if row["id"] in ids or not row["input"].strip():
            raise ValueError("Duplicate or empty evaluation case")
        ids.add(row["id"])
        if train_phrases & phrase_windows(row["input"]):
            raise ValueError(f"Evaluation input overlaps training: {row['id']}")
        if row["kind"] == "novel":
            continue
        p = passages[row["passage_id"]]
        audit = row["review"]
        if (
            p["split"] != "eval"
            or pieces[p["source"]]["split"] != "eval"
            or row["source"] != p["source"]
            or row["source"] in train_sources
            or row["reference"] != p["target"]
            or audit["verdict"] != "pass"
            or audit["input_sha256"] != text_sha(row["input"])
            or audit["reference_sha256"] != text_sha(row["reference"])
            or check(row["reference"], row["input"])["flags"]
            or train_phrases & phrase_windows(row["reference"])
        ):
            raise ValueError(f"Invalid held-out case: {row['id']}")
    return cases


def prepare(dataset, work, additions, novel, config_path, baseline_prompt, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh experiment directory to preserve frozen data")
    config = load_config(config_path)
    policy = json.loads((dataset / "policy.json").read_text())
    errors = validate_pairs_file(
        dataset / "pairs.jsonl", system_prompt=policy["system_prompt"]
    )
    if errors:
        raise ValueError(f"Invalid training corpus: {errors[0]}")
    corpus = assemble_training_corpus(
        dataset / "pairs.jsonl",
        system_prompt=policy["system_prompt"],
        val_frac=config["validation_fraction"],
        seed=config["seed"],
    )
    for row in corpus.train + corpus.val:
        quality = row["meta"]["quality_checks"]
        review = quality.get("fidelity") or quality.get("legacy_review")
        if (
            not review
            or review["verdict"] != "pass"
            or review["draft_sha256"] != text_sha(row["messages"][1]["content"])
        ):
            raise ValueError("Every training example needs an exact semantic approval")
    cases = build_cases(dataset, work, additions, novel)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output / "train.jsonl", [{"messages": r["messages"]} for r in corpus.train]
    )
    write_jsonl(
        output / "validation.jsonl", [{"messages": r["messages"]} for r in corpus.val]
    )
    write_jsonl(output / "evaluation.jsonl", cases)
    write_json(output / "config.json", config)
    write_json(output / "policy.json", policy)
    (output / "baseline-style.txt").write_text(
        baseline_prompt.read_text().strip() + "\n"
    )
    write_json(
        output / "splits.json",
        {
            split: [
                {
                    "id": r["meta"]["passage_id"],
                    "source": r["meta"]["source"],
                    "target_sha256": r["meta"]["target_sha256"],
                }
                for r in rows
            ]
            for split, rows in (("train", corpus.train), ("validation", corpus.val))
        },
    )
    names = [
        "train.jsonl",
        "validation.jsonl",
        "evaluation.jsonl",
        "config.json",
        "policy.json",
        "baseline-style.txt",
        "splits.json",
    ]
    manifest = {
        "schema_version": 1,
        "files": {name: file_sha(output / name) for name in names},
        "source_training_sha256": file_sha(dataset / "pairs.jsonl"),
        "original_evaluation_sha256": file_sha(dataset / "evaluation" / "pairs.jsonl"),
        "training_examples": corpus.n_train,
        "validation_examples": corpus.n_val,
        "evaluation_cases": len(cases),
        "reference_cases": sum(r["kind"] == "reference" for r in cases),
        "novel_cases": sum(r["kind"] == "novel" for r in cases),
        "steps_per_epoch": math.ceil(corpus.n_train / config["batch_size"]),
        "total_steps": math.ceil(corpus.n_train / config["batch_size"])
        * config["epochs"],
        "loss_scope": "last_assistant_message",
        "held_out_by_whole_piece": True,
        "paid_calls_made": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("data/eb-white"))
    p.add_argument("--work", type=Path, default=Path("work/white-combined"))
    p.add_argument(
        "--additions", type=Path, default=Path("work/experiment-v1/heldout-cases.jsonl")
    )
    p.add_argument(
        "--novel", type=Path, default=Path("examples/evaluation-inputs.jsonl")
    )
    p.add_argument("--config", type=Path, default=Path("configs/qwen3.5-9b-v1.json"))
    p.add_argument(
        "--baseline-prompt", type=Path, default=Path("prompts/baseline-style.txt")
    )
    p.add_argument("--output", type=Path, default=Path("data/tinker/white-v1"))
    a = p.parse_args()
    manifest = prepare(
        a.dataset, a.work, a.additions, a.novel, a.config, a.baseline_prompt, a.output
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
