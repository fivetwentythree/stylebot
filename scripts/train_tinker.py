"""Launch the real Tinker Cookbook SFT run from a verified prepared experiment."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from experiment import (
    completed_samples,
    file_sha,
    pinned_tokenizer,
    positive_budget,
    sample_identity,
    verify_prepared,
    write_json,
)
from review_batch import read_jsonl


def check_baselines(prepared, run_dir):
    cases = read_jsonl(prepared / "evaluation.jsonl")
    for arm in ("base-plain", "base-style"):
        completed_samples(
            run_dir / "baselines" / f"{arm}.jsonl",
            cases,
            sample_identity(prepared, arm),
            require_all=True,
        )


def make_config(prepared, run_dir):
    from tinker_cookbook.supervised.train import Config
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_dataset import FrozenChatBuilder

    values = json.loads((prepared / "config.json").read_text())
    report = json.loads((prepared / "token-report.json").read_text())
    frozen = json.loads((prepared / "manifest.json").read_text())
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=values["model"],
        renderer_name=values["renderer"],
        max_length=values["max_length"],
        batch_size=values["batch_size"],
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    builder = FrozenChatBuilder(
        prepared_dir=str(prepared.resolve()),
        shuffle_seed=values["seed"],
        common_config=common,
    )
    return Config(
        log_path=str((run_dir / "tinker").resolve()),
        model_name=values["model"],
        recipe_name="stylebot_white_sft",
        session_name="StyleBot White v1",
        renderer_name=values["renderer"],
        dataset_builder=builder,
        learning_rate=report["learning_rate"],
        lr_schedule=values["lr_schedule"],
        num_epochs=values["epochs"],
        lora_rank=values["lora_rank"],
        eval_every=values["eval_every"],
        save_every=values["save_every"],
        max_steps=frozen["total_steps"],
        submit_ahead=0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=Path("data/tinker/white-v1"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/white-v1"))
    p.add_argument("--max-cost-usd", type=positive_budget, required=True)
    a = p.parse_args()
    if not os.environ.get("TINKER_API_KEY"):
        p.error("Set TINKER_API_KEY locally before launching the paid Tinker run.")
    verify_prepared(a.prepared, token_report=True)
    manifest_path = a.run_dir / "run.json"
    prepared_sha = file_sha(a.prepared / "manifest.json")
    report_sha = file_sha(a.prepared / "token-report.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (
            previous["data"]["prepared_sha256"] != prepared_sha
            or previous["data"].get("token_report_sha256") != report_sha
        ):
            p.error(
                "This run directory belongs to a different experiment or token report."
            )
        if previous.get("status") == "complete":
            print(
                f"Training is already complete; checkpoint is saved in {manifest_path}"
            )
            return
    report = json.loads((a.prepared / "token-report.json").read_text())
    estimated = report["cost_estimates_usd"]["training_and_scheduled_validation"]
    if estimated > a.max_cost_usd:
        p.error(
            f"Estimated token charge ${estimated:.3f} exceeds the supplied allowance."
        )
    check_baselines(a.prepared, a.run_dir)
    pinned_tokenizer(a.prepared)
    config = make_config(a.prepared, a.run_dir)
    values = json.loads((a.prepared / "config.json").read_text())
    policy = json.loads((a.prepared / "policy.json").read_text())
    records = read_jsonl(a.prepared / "train.jsonl") + read_jsonl(
        a.prepared / "validation.jsonl"
    )
    manifest = {
        "hyperparameters": {
            "base_model": values["model"],
            "renderer_name": values["renderer"],
            "learning_rate": report["learning_rate"],
            "lora_rank": values["lora_rank"],
            "batch_size": values["batch_size"],
            "epochs": values["epochs"],
        },
        "data": {
            "style_system": policy["system_prompt"],
            "prepared_sha256": prepared_sha,
            "token_report_sha256": report_sha,
            "chunk_scale": {
                "budgets": {"None": len(records)},
                "modes": {"None": len(records)},
                "max_target_chars": max(
                    len(r["messages"][2]["content"]) for r in records
                ),
            },
        },
        "baseline_sha256": {
            arm: file_sha(a.run_dir / "baselines" / f"{arm}.jsonl")
            for arm in ("base-plain", "base-style")
        },
        "token_charge_estimate_usd": estimated,
        "status": "training",
    }
    write_json(manifest_path, manifest)
    from tinker_cookbook.supervised import train
    from tinker_cookbook import checkpoint_utils

    asyncio.run(train.main(config))
    checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
    if not checkpoint or not checkpoint.sampler_path:
        raise RuntimeError("Training returned without a sampler checkpoint")
    manifest["result"] = {
        "renderer_name": values["renderer"],
        "checkpoints": {
            "sampler_path": checkpoint.sampler_path,
            "state_path": checkpoint.state_path,
        },
    }
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"Trained checkpoint saved in {manifest_path}")


if __name__ == "__main__":
    main()
