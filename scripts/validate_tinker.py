"""Tokenize frozen chats with the official Tinker renderer; no Tinker API calls."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from experiment import ARMS, arm_prompt, file_sha, verify_prepared, write_json
from review_batch import read_jsonl, write_jsonl
from tinker_dataset import decode_datum, encode_datum


def validate(root):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from tinker_cookbook import hyperparam_utils, renderers
    from tinker_cookbook.supervised.data import conversation_to_datum

    manifest = verify_prepared(root)
    config = json.loads((root / "config.json").read_text())
    existing_path = root / "token-report.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
    if existing:
        verify_prepared(root, token_report=True)
    snapshot = snapshot_download(
        repo_id=config["model"],
        revision=existing.get("tokenizer_revision"),
        cache_dir=str(Path("work/huggingface/hub").resolve()),
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
        ],
    )
    revision = Path(snapshot).name
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    renderer = renderers.get_renderer(config["renderer"], tokenizer)
    lr = (
        config["learning_rate"]
        or existing.get("learning_rate")
        or hyperparam_utils.get_lr(config["model"])
    )
    report = {
        "manifest_sha256": file_sha(root / "manifest.json"),
        "model": config["model"],
        "renderer": config["renderer"],
        "tokenizer_revision": revision,
        "learning_rate": lr,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("tinker", "tinker-cookbook", "transformers", "tokenizers")
        },
        "loss_scope": "last_assistant_message",
        "train": {},
        "validation": {},
        "generation_prompt_tokens": {},
        "tokenized_files": {},
        "paid_calls_made": False,
    }
    for split in ("train", "validation"):
        lengths, target_counts, datums = [], [], []
        for row in read_jsonl(root / f"{split}.jsonl"):
            messages = row["messages"]
            if [m["role"] for m in messages] != ["system", "user", "assistant"]:
                raise ValueError("Expected exactly system, user and assistant turns")
            model_input, weights = renderer.build_supervised_example(
                messages,
                train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            prompt = renderer.build_generation_prompt(messages[:2])
            tokens = model_input.to_ints()
            if tokens[: prompt.length] != prompt.to_ints():
                raise ValueError("Training and generation templates disagree")
            if len(tokens) > config["max_length"]:
                raise ValueError("An example would be truncated; increase max_length")
            raw_weights = weights.tolist()
            if any(raw_weights[: prompt.length]) or not any(
                raw_weights[prompt.length :]
            ):
                raise ValueError(
                    "Loss mask includes prompt tokens or excludes the target"
                )
            datum = conversation_to_datum(
                messages,
                renderer,
                max_length=config["max_length"],
                train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            # Ensure the actual SDK object accepts its serialized representation.
            serialized = encode_datum(datum)
            restored = decode_datum(serialized)
            if restored.model_input.to_ints() != datum.model_input.to_ints():
                raise ValueError("SDK datum roundtrip changed tokens")
            if encode_datum(restored) != serialized:
                raise ValueError("SDK datum roundtrip changed targets or loss weights")
            datums.append(serialized)
            lengths.append(datum.model_input.length)
            target_counts.append(sum(w > 0 for w in raw_weights))
        path = root / "tokenized" / f"{split}.jsonl"
        write_jsonl(path, datums)
        report["tokenized_files"][f"tokenized/{split}.jsonl"] = file_sha(path)
        report[split] = {
            "examples": len(lengths),
            "tokens": sum(lengths),
            "assistant_tokens": sum(target_counts),
            "max_sequence_tokens": max(lengths),
        }
    cases = read_jsonl(root / "evaluation.jsonl")
    for arm in ARMS:
        report["generation_prompt_tokens"][arm] = {
            case["id"]: renderer.build_generation_prompt(
                [
                    {"role": "system", "content": arm_prompt(root, arm)},
                    {"role": "user", "content": case["input"]},
                ]
            ).length
            for case in cases
        }
    # The cookbook evaluates at step 0 and each configured interval thereafter.
    evaluations = math.ceil(manifest["total_steps"] / config["eval_every"])
    prices = config["price_per_million_tokens"]
    training_tokens = report["train"]["tokens"] * config["epochs"]
    validation_tokens = report["validation"]["tokens"] * evaluations
    sample_output_max = len(cases) * config["sampling"]["max_tokens"]
    sampling = {
        arm: (
            sum(counts.values()) * prices["prefill"]
            + sample_output_max * prices["sample"]
        )
        / 1e6
        for arm, counts in report["generation_prompt_tokens"].items()
    }
    report["cost_estimates_usd"] = {
        "training_and_scheduled_validation": (training_tokens + validation_tokens)
        * prices["train"]
        / 1e6,
        "sampling_ceiling_by_arm": sampling,
        "all_three_sampling_arms_ceiling": sum(sampling.values()),
        "basis": "Uncached listed token rates; excludes checkpoint storage and provider billing changes.",
    }
    write_json(root / "token-report.json", report)
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=Path("data/tinker/white-v1"))
    validate(p.parse_args().prepared)


if __name__ == "__main__":
    main()
