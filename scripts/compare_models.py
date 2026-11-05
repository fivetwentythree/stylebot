"""Capture matched Tinker samples and make a blinded review packet."""

import argparse
import json
import os
import random
from pathlib import Path

from experiment import (
    ARMS,
    arm_prompt,
    completed_samples,
    file_sha,
    pinned_tokenizer,
    positive_budget,
    sample_identity,
    sampling_ceiling,
    verify_prepared,
    write_json,
)
from review_batch import read_jsonl, text_sha, write_jsonl


def sample(prepared, run_dir, arm, allowance):
    verify_prepared(prepared, token_report=True)
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("Set TINKER_API_KEY locally before paid sampling.")
    cases = read_jsonl(prepared / "evaluation.jsonl")
    config = json.loads((prepared / "config.json").read_text())
    checkpoint = None
    if arm == "tuned":
        run = json.loads((run_dir / "run.json").read_text())
        if (
            run["data"]["prepared_sha256"] != file_sha(prepared / "manifest.json")
            or run["data"].get("token_report_sha256")
            != file_sha(prepared / "token-report.json")
            or run.get("status") != "complete"
        ):
            raise ValueError("Checkpoint and evaluation experiment do not match")
        checkpoint = run["result"]["checkpoints"]["sampler_path"]
    identity = sample_identity(prepared, arm, checkpoint)
    folder = "outputs" if arm == "tuned" else "baselines"
    out = run_dir / folder / f"{arm}.jsonl"
    done = completed_samples(out, cases, identity)
    remaining = {c["id"] for c in cases} - done.keys()
    estimate = sampling_ceiling(prepared, arm, remaining)
    if estimate > allowance:
        raise ValueError(
            f"Sampling ceiling ${estimate:.3f} exceeds the supplied allowance"
        )
    if not remaining:
        print(f"{arm}: all {len(cases)} outputs already saved")
        return
    import tinker
    from tinker import types
    from tinker_cookbook import renderers

    tokenizer = pinned_tokenizer(prepared)
    renderer = renderers.get_renderer(config["renderer"], tokenizer)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tinker.ServiceClient() as service:
        client = service.create_sampling_client(
            model_path=checkpoint,
            base_model=None if checkpoint else config["model"],
        )
        with out.open("a", encoding="utf-8") as stream:
            for position, case in enumerate(cases):
                if case["id"] in done:
                    continue
                prompt = renderer.build_generation_prompt(
                    [
                        {"role": "system", "content": arm_prompt(prepared, arm)},
                        {"role": "user", "content": case["input"]},
                    ]
                )
                params = types.SamplingParams(
                    max_tokens=config["sampling"]["max_tokens"],
                    temperature=config["sampling"]["temperature"],
                    seed=config["sampling"]["seed"] + position,
                    stop=renderer.get_stop_sequences(),
                )
                result = client.sample(
                    prompt, num_samples=1, sampling_params=params
                ).result()
                sequence = result.sequences[0]
                message, termination = renderer.parse_response(sequence.tokens)
                text = message["content"]
                if (
                    not isinstance(text, str)
                    or not text.strip()
                    or not termination.is_clean
                ):
                    raise ValueError(
                        f"Incomplete output for {case['id']}; prior outputs are saved"
                    )
                row = {
                    "id": case["id"],
                    "run": identity,
                    "input_sha256": text_sha(case["input"]),
                    "output": text.strip(),
                    "clean_termination": True,
                    "input_tokens": prompt.length,
                    "output_tokens": len(sequence.tokens),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{arm}: saved {position + 1}/{len(cases)}")


def packet(prepared, run_dir):
    verify_prepared(prepared, token_report=True)
    cases = read_jsonl(prepared / "evaluation.jsonl")
    run = json.loads((run_dir / "run.json").read_text())
    if (
        run["data"]["prepared_sha256"] != file_sha(prepared / "manifest.json")
        or run["data"].get("token_report_sha256")
        != file_sha(prepared / "token-report.json")
        or run.get("status") != "complete"
    ):
        raise ValueError("Trained run does not match this prepared experiment")
    checkpoint = run["result"]["checkpoints"]["sampler_path"]
    outputs = {}
    for arm in ARMS:
        folder = "outputs" if arm == "tuned" else "baselines"
        outputs[arm] = completed_samples(
            run_dir / folder / f"{arm}.jsonl",
            cases,
            sample_identity(prepared, arm, checkpoint if arm == "tuned" else None),
            require_all=True,
        )
    lines = [
        "# Blind rewrite comparison",
        "",
        "Use the review rubric before opening blind-key.json.",
        "",
    ]
    mapping, ratings = {}, []
    rng = random.Random(523)
    for case in cases:
        arms = list(ARMS)
        rng.shuffle(arms)
        mapping[case["id"]] = dict(zip("ABC", arms))
        lines += [
            f"## {case['id']} — {case['title']}",
            "",
            "### Input",
            "",
            case["input"],
            "",
        ]
        for fact in case.get("fact_checks", []):
            lines.append("- " + fact)
        lines.append("")
        for letter, arm in mapping[case["id"]].items():
            lines += [
                f"### Candidate {letter}",
                "",
                outputs[arm][case["id"]]["output"],
                "",
            ]
        ratings.append(
            {
                "id": case["id"],
                "candidates": {
                    letter: {"style_1_to_5": None, "fidelity_pass": None, "note": ""}
                    for letter in "ABC"
                },
                "preferred": None,
            }
        )
    review = run_dir / "review"
    review.mkdir(parents=True, exist_ok=True)
    if (review / "ratings.jsonl").exists():
        raise ValueError("Ratings already exist; refusing to overwrite review work")
    (review / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(review / "blind-key.json", mapping)
    write_jsonl(review / "ratings.jsonl", ratings)
    print(f"Saved blinded packet in {review}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--arm", choices=ARMS, required=True)
    sampling.add_argument("--max-cost-usd", type=positive_budget, required=True)
    sub.add_parser("packet")
    for mode in sub.choices.values():
        mode.add_argument("--prepared", type=Path, default=Path("data/tinker/white-v1"))
        mode.add_argument("--run-dir", type=Path, default=Path("runs/white-v1"))
    a = p.parse_args()
    if a.command == "sample":
        sample(a.prepared, a.run_dir, a.arm, a.max_cost_usd)
    else:
        packet(a.prepared, a.run_dir)


if __name__ == "__main__":
    main()
