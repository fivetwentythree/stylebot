"""Offline regressions for batching, experiment isolation and matched comparisons."""

import copy
import importlib.metadata
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_models import packet, sample
from experiment import (
    ARMS,
    completed_samples,
    file_sha,
    sample_identity,
    verify_prepared,
    write_json,
)
from prepare_experiment import build_cases
from review_batch import read_jsonl, text_sha, write_jsonl
from tinker_dataset import CompleteBatches, decode_datum, encode_datum


class BatchTests(unittest.TestCase):
    def test_partial_batch_keeps_every_example_once(self):
        dataset = CompleteBatches(range(162), 8, seed=17)
        for epoch in range(3):
            dataset.set_epoch(epoch)
            batches = [dataset.get_batch(i) for i in range(len(dataset))]
            self.assertEqual(len(batches), 21)
            self.assertEqual(len(batches[-1]), 2)
            self.assertEqual(
                sorted(x for batch in batches for x in batch), list(range(162))
            )
        with self.assertRaises(IndexError):
            dataset.get_batch(21)

    def test_resume_reconstructs_the_same_epoch_order(self):
        first, resumed = (
            CompleteBatches(range(41), 8, 12),
            CompleteBatches(range(41), 8, 12),
        )
        first.set_epoch(2)
        resumed.set_epoch(2)
        self.assertEqual(first.order, resumed.order)
        resumed.set_epoch(3)
        self.assertNotEqual(first.order, resumed.order)

    def test_sdk_serialization_preserves_targets_and_loss_mask(self):
        import tinker
        from tinker.types import ModelInput, TensorData

        original = tinker.Datum(
            model_input=ModelInput.from_ints([11, 12, 13]),
            loss_fn_inputs={
                "target_tokens": TensorData(
                    data=[12, 13, 14], dtype="int64", shape=[3]
                ),
                "weights": TensorData(data=[0.0, 0.0, 1.0], dtype="float32", shape=[3]),
            },
        )
        restored = decode_datum(json.loads(json.dumps(encode_datum(original))))
        self.assertEqual(restored.model_input.to_ints(), [11, 12, 13])
        self.assertEqual(encode_datum(restored), encode_datum(original))


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.prepared, self.run = self.root / "prepared", self.root / "run"
        config_path = Path(__file__).resolve().parents[1] / "configs/qwen3.5-9b-v1.json"
        write_json(self.prepared / "config.json", json.loads(config_path.read_text()))
        write_json(
            self.prepared / "policy.json", {"system_prompt": "Rewrite faithfully."}
        )
        (self.prepared / "baseline-style.txt").write_text(
            "Use the specified prose style.\n"
        )
        self.cases = [
            {
                "id": "novel-001",
                "kind": "novel",
                "source": "invented/test",
                "title": "A parcel",
                "input": "I collected a parcel at six.",
                "fact_checks": ["Collection is at six."],
            }
        ]
        write_jsonl(self.prepared / "evaluation.jsonl", self.cases)
        names = ("config.json", "policy.json", "baseline-style.txt", "evaluation.jsonl")
        write_json(
            self.prepared / "manifest.json",
            {
                "files": {name: file_sha(self.prepared / name) for name in names},
            },
        )
        write_json(
            self.prepared / "token-report.json",
            {
                "manifest_sha256": file_sha(self.prepared / "manifest.json"),
                "tokenized_files": {},
                "versions": {"tinker": importlib.metadata.version("tinker")},
                "generation_prompt_tokens": {arm: {"novel-001": 100} for arm in ARMS},
            },
        )

    def output(self, arm, checkpoint=None):
        return {
            "id": "novel-001",
            "run": sample_identity(self.prepared, arm, checkpoint),
            "input_sha256": text_sha(self.cases[0]["input"]),
            "output": "At six I collected a parcel.",
            "clean_termination": True,
        }

    def test_frozen_inputs_and_token_data_cannot_change_silently(self):
        verify_prepared(self.prepared, token_report=True)
        (self.prepared / "baseline-style.txt").write_text("A changed instruction")
        with self.assertRaisesRegex(ValueError, "Frozen experiment file changed"):
            verify_prepared(self.prepared, token_report=True)

    def test_tokenized_file_changes_are_detected(self):
        path = self.prepared / "tokenized/train.jsonl"
        write_jsonl(path, [{"test": "original"}])
        report_path = self.prepared / "token-report.json"
        report = json.loads(report_path.read_text())
        report["tokenized_files"] = {"tokenized/train.jsonl": file_sha(path)}
        write_json(report_path, report)
        verify_prepared(self.prepared, token_report=True)
        write_jsonl(path, [{"test": "changed"}])
        with self.assertRaisesRegex(ValueError, "Tokenized data changed"):
            verify_prepared(self.prepared, token_report=True)

    def test_comparison_resume_refuses_other_prompts_seeds_checkpoints_or_inputs(self):
        path = self.run / "sample.jsonl"
        identity = sample_identity(self.prepared, "tuned", "tinker://test/sampler/one")
        original = self.output("tuned", "tinker://test/sampler/one")
        changes = [
            ("prompt_sha256", "changed"),
            ("checkpoint", "tinker://test/sampler/two"),
            ("sampling", {"seed": 999}),
            ("token_report_sha256", "changed"),
        ]
        for key, value in changes:
            row = copy.deepcopy(original)
            row["run"][key] = value
            write_jsonl(path, [row])
            with self.subTest(key=key), self.assertRaises(ValueError):
                completed_samples(path, self.cases, identity)
        row = copy.deepcopy(original)
        row["input_sha256"] = "changed"
        write_jsonl(path, [row])
        with self.assertRaises(ValueError):
            completed_samples(path, self.cases, identity)

    def test_duplicate_and_incomplete_results_cannot_count_as_complete(self):
        path = self.run / "sample.jsonl"
        row = self.output("base-plain")
        identity = sample_identity(self.prepared, "base-plain")
        for rows in ([row, row], [row | {"clean_termination": False}], []):
            write_jsonl(path, rows)
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                completed_samples(path, self.cases, identity, require_all=True)

    def test_missing_key_or_inadequate_allowance_never_opens_a_client(self):
        with patch("tinker.ServiceClient") as client:
            with (
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(ValueError, "API_KEY"),
            ):
                sample(self.prepared, self.run, "base-plain", 1.0)
            with patch.dict(os.environ, {"TINKER_API_KEY": "offline-test-only"}):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    sample(self.prepared, self.run, "base-plain", 0.00001)
            client.assert_not_called()

    def test_blinded_packet_keeps_the_key_separate_and_preserves_existing_ratings(self):
        checkpoint = "tinker://offline-test/sampler/final"
        write_json(
            self.run / "run.json",
            {
                "status": "complete",
                "data": {
                    "prepared_sha256": file_sha(self.prepared / "manifest.json"),
                    "token_report_sha256": file_sha(
                        self.prepared / "token-report.json"
                    ),
                },
                "result": {"checkpoints": {"sampler_path": checkpoint}},
            },
        )
        for arm in ARMS:
            folder = "outputs" if arm == "tuned" else "baselines"
            write_jsonl(
                self.run / folder / f"{arm}.jsonl",
                [
                    self.output(arm, checkpoint if arm == "tuned" else None),
                ],
            )
        with redirect_stdout(StringIO()):
            packet(self.prepared, self.run)
        comparison = (self.run / "review/comparison.md").read_text()
        for arm in ARMS:
            self.assertNotIn(arm, comparison)
        key = json.loads((self.run / "review/blind-key.json").read_text())
        self.assertEqual(set(key["novel-001"].values()), set(ARMS))
        ratings = self.run / "review/ratings.jsonl"
        before = ratings.read_bytes()
        with self.assertRaisesRegex(ValueError, "Ratings already exist"):
            packet(self.prepared, self.run)
        self.assertEqual(ratings.read_bytes(), before)


class EvaluationIsolationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.dataset, self.work = root / "data", root / "work"
        self.additions, self.novel = root / "additions.jsonl", root / "novel.jsonl"
        self.training_text = " ".join(f"word{i}" for i in range(30))
        write_jsonl(
            self.dataset / "pairs.jsonl",
            [
                {
                    "meta": {"source": "training-piece"},
                    "messages": [
                        {"content": "Instruction"},
                        {"content": self.training_text},
                        {"content": "Training target"},
                    ],
                }
            ],
        )
        write_jsonl(
            self.dataset / "evaluation/pairs.jsonl",
            [{"meta": {"passage_id": "old-01"}}],
        )
        write_jsonl(self.dataset / "evaluation/approved-pairs.jsonl", [])
        self.reference = {
            "id": "white-new-01",
            "kind": "reference",
            "passage_id": "new-01",
            "source": "held-out-piece",
            "input": "A new ordinary draft.",
            "reference": "A target passage.",
            "review": {
                "verdict": "pass",
                "input_sha256": text_sha("A new ordinary draft."),
                "reference_sha256": text_sha("A target passage."),
            },
        }
        write_jsonl(
            self.work / "pieces.jsonl", [{"source": "held-out-piece", "split": "eval"}]
        )
        write_jsonl(
            self.work / "passages.jsonl",
            [
                {
                    "id": "new-01",
                    "source": "held-out-piece",
                    "split": "eval",
                    "target": "A target passage.",
                }
            ],
        )
        write_jsonl(self.additions, [self.reference])
        write_jsonl(
            self.novel,
            [
                {
                    "id": "novel-test",
                    "kind": "novel",
                    "input": "An invented input.",
                    "fact_checks": ["Invented only."],
                }
            ],
        )

    def build(self):
        # Semantic review and mechanical checks are covered separately; isolate split integrity here.
        with patch("prepare_experiment.check", return_value={"flags": []}):
            return build_cases(self.dataset, self.work, self.additions, self.novel)

    def test_an_entire_training_piece_cannot_be_used_as_evaluation(self):
        self.assertEqual(len(self.build()), 2)
        rows = read_jsonl(self.dataset / "pairs.jsonl")
        rows[0]["meta"]["source"] = "held-out-piece"
        write_jsonl(self.dataset / "pairs.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Invalid held-out case"):
            self.build()

    def test_evaluation_cannot_reuse_training_phrases(self):
        self.assertEqual(len(self.build()), 2)
        write_jsonl(
            self.novel,
            [
                {
                    "id": "novel-test",
                    "kind": "novel",
                    "input": self.training_text,
                    "fact_checks": ["Overlapping input."],
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            self.build()

    def test_new_cases_cannot_replace_original_frozen_cases(self):
        self.assertEqual(len(self.build()), 2)
        write_jsonl(self.additions, [self.reference | {"passage_id": "old-01"}])
        with self.assertRaisesRegex(ValueError, "must not replace"):
            self.build()


if __name__ == "__main__":
    unittest.main()
