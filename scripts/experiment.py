"""Shared integrity checks for a frozen Tinker experiment."""

import importlib.metadata
import json
import math
from hashlib import sha256
from pathlib import Path

from review_batch import read_jsonl, text_sha

ARMS = ("base-plain", "base-style", "tuned")


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return text_sha(json.dumps(value, sort_keys=True, ensure_ascii=False))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "batch_size",
        "epochs",
        "max_length",
        "lora_rank",
        "eval_every",
        "save_every",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if config["learning_rate"] is not None:
        positive_budget(config["learning_rate"])
    if config["lr_schedule"] not in {"linear", "cosine", "constant"}:
        raise ValueError("Unknown learning-rate schedule")
    if (
        config["model"] != "Qwen/Qwen3.5-9B"
        or config["renderer"] != "qwen3_5_disable_thinking"
    ):
        raise ValueError("This experiment is configured for non-thinking Qwen3.5-9B")
    sampling = config["sampling"]
    if type(sampling["max_tokens"]) is not int or sampling["max_tokens"] <= 0:
        raise ValueError("Sampling max_tokens must be positive")
    if not math.isfinite(sampling["temperature"]) or sampling["temperature"] < 0:
        raise ValueError("Invalid sampling temperature")
    for key in ("train", "prefill", "sample"):
        positive_budget(config["price_per_million_tokens"][key])
    return config


def positive_budget(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("The spending allowance must be a finite positive number")
    return number


def verify_prepared(root, *, token_report=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha(path) != expected:
            raise ValueError(f"Frozen experiment file changed: {name}")
    if token_report:
        report = json.loads((root / "token-report.json").read_text())
        if report["manifest_sha256"] != file_sha(root / "manifest.json"):
            raise ValueError("Token report belongs to a different experiment")
        for name, expected in report["tokenized_files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or file_sha(path) != expected:
                raise ValueError(f"Tokenized data changed: {name}")
        for package, expected in report["versions"].items():
            if importlib.metadata.version(package) != expected:
                raise ValueError(
                    f"SDK environment changed: {package}; use uv sync --locked"
                )
    return manifest


def arm_prompt(root, arm):
    if arm not in ARMS:
        raise ValueError("Unknown comparison arm")
    root = Path(root)
    if arm == "base-style":
        return (root / "baseline-style.txt").read_text().strip()
    return json.loads((root / "policy.json").read_text())["system_prompt"]


def sample_identity(root, arm, checkpoint=None):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    return {
        "arm": arm,
        "model": config["model"],
        "renderer": config["renderer"],
        "checkpoint": checkpoint,
        "experiment_sha256": file_sha(root / "manifest.json"),
        "token_report_sha256": file_sha(root / "token-report.json"),
        "prompt_sha256": text_sha(arm_prompt(root, arm)),
        "sampling": config["sampling"],
    }


def completed_samples(path, cases, identity, *, require_all=False):
    path = Path(path)
    rows = read_jsonl(path) if path.exists() else []
    expected = {case["id"]: case for case in cases}
    seen = {}
    for row in rows:
        pid = row["id"]
        if (
            pid in seen
            or pid not in expected
            or row.get("run") != identity
            or row.get("input_sha256") != text_sha(expected[pid]["input"])
            or not row.get("output", "").strip()
            or not row.get("clean_termination")
        ):
            raise ValueError(f"Stale, duplicate or incomplete sampling result: {pid}")
        seen[pid] = row
    if require_all and seen.keys() != expected.keys():
        raise ValueError(f"Comparison arm incomplete: {path}")
    return seen


def sampling_ceiling(root, arm, case_ids=None):
    root = Path(root)
    report = json.loads((root / "token-report.json").read_text())
    config = json.loads((root / "config.json").read_text())
    by_case = report["generation_prompt_tokens"][arm]
    ids = set(by_case) if case_ids is None else set(case_ids)
    prices = config["price_per_million_tokens"]
    return (
        sum(by_case[pid] for pid in ids) * prices["prefill"]
        + len(ids) * config["sampling"]["max_tokens"] * prices["sample"]
    ) / 1_000_000


def pinned_tokenizer(root):
    from transformers import AutoTokenizer
    from tinker_cookbook.tokenizer_utils import register_tokenizer

    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    report = json.loads((root / "token-report.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        revision=report["tokenizer_revision"],
        cache_dir=str(Path("work/huggingface/hub").resolve()),
        local_files_only=True,
    )
    register_tokenizer(config["model"], lambda: tokenizer)
    return tokenizer
