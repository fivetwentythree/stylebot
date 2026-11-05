"""Regression checks for review integrity, split isolation and prompt propagation."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from package import apply_fidelity_review
from review_batch import (
    record_decisions,
    select_passages,
    text_sha,
    with_neighbors,
    write_jsonl,
)
from rewrite import serving_options
from stylebot import backends, draws
from stylebot.pairs import system_sha, validate_pairs_file


class FidelityReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.passage = {
            "id": "sample-01",
            "split": "train",
            "target": "I walked home in the rain.",
        }
        self.passage["target_sha256"] = text_sha(self.passage["target"])
        self.draft = {"id": "sample-01", "text": "I went home while it was raining."}
        self.decision = {
            "id": "sample-01",
            "target_sha256": self.passage["target_sha256"],
            "draft_sha256": text_sha(self.draft["text"]),
            "verdict": "pass",
            "reviewer": "test reviewer",
            "note": "Preserves the narrator, journey and weather.",
        }

    def record(self, *, decisions=None):
        paths = [self.root / name for name in ("passages", "drafts", "decisions")]
        for path, rows in zip(
            paths,
            ([self.passage], [self.draft], decisions or [self.decision]),
        ):
            write_jsonl(path, rows)
        return list(record_decisions(*paths))

    def test_exact_review_is_accepted_and_bound_to_pair(self):
        row = self.record()[0]
        result = {"flags": []}
        apply_fidelity_review(self.passage, row, result)
        self.assertEqual(result["flags"], [])
        self.assertEqual(row["fidelity"]["draft_sha256"], text_sha(row["text"]))

    def test_changed_draft_invalidates_review(self):
        self.draft["text"] += " I bought an umbrella."
        with self.assertRaisesRegex(ValueError, "Stale fidelity"):
            self.record()

    def test_changed_target_invalidates_review(self):
        self.passage["target"] = "I walked home in the snow."
        self.passage["target_sha256"] = text_sha(self.passage["target"])
        with self.assertRaisesRegex(ValueError, "Stale fidelity"):
            self.record()

    def test_evaluation_target_cannot_be_approved(self):
        self.passage["split"] = "eval"
        with self.assertRaisesRegex(ValueError, "Invalid training target"):
            self.record()

    def test_failed_generation_can_only_be_rejected(self):
        self.draft["text"] = ""
        self.decision["draft_sha256"] = text_sha("")
        with self.assertRaisesRegex(ValueError, "Incomplete fidelity"):
            self.record()
        self.decision["verdict"] = "reject"
        self.assertEqual(self.record()[0]["fidelity"]["verdict"], "reject")

    def test_duplicate_decisions_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Duplicate passage ID"):
            self.record(decisions=[self.decision, self.decision])

    def test_packager_rechecks_hash_and_split(self):
        row = self.record()[0]
        row["text"] += " Changed."
        result = {"flags": []}
        apply_fidelity_review(self.passage | {"split": "eval"}, row, result)
        self.assertIn("stale_fidelity_review", result["flags"])
        self.assertIn("evaluation_split_locked", result["flags"])

    def test_neighbor_context_stays_outside_target(self):
        result = with_neighbors(
            self.passage,
            {
                "paragraphs": [
                    "Earlier context.",
                    self.passage["target"],
                    "Later context.",
                ]
            },
            words=2,
        )
        self.assertEqual(result["target"], self.passage["target"])
        self.assertEqual(result["context_before"], "Earlier context.")
        self.assertEqual(result["context_after"], "Later context.")

    def test_selection_excludes_entire_held_out_source(self):
        work, data = self.root / "work", self.root / "data"
        rows = [
            self.passage | {"source": "book/safe.md", "words": 7, "chunk_index": 0},
            self.passage
            | {
                "id": "mislabelled",
                "source": "book/held.md",
                "words": 7,
                "chunk_index": 0,
            },
        ]
        write_jsonl(work / "passages.jsonl", rows)
        write_jsonl(
            work / "pieces.jsonl",
            [
                {
                    "source": row["source"],
                    "split": "train",
                    "paragraphs": [row["target"]],
                }
                for row in rows
            ],
        )
        write_jsonl(data / "pairs.jsonl", [])
        write_jsonl(
            data / "evaluation" / "pairs.jsonl", [{"meta": {"source": "book/held.md"}}]
        )
        write_jsonl(data / "passages.jsonl", [])
        selected = select_passages(work, data, 10)
        self.assertEqual([row["source"] for row in selected], ["book/safe.md"])


class ServingPromptTests(unittest.TestCase):
    def test_policy_view_carries_author_free_prompt_to_inference(self):
        repo = Path(__file__).resolve().parents[1]
        policy = draws.load_policy(repo / "examples" / "policy.json")
        record = json.loads((repo / "examples" / "pairs.jsonl").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "inputs" / "accepted.jsonl",
                [draws.draw_from_pair(copy.deepcopy(record)).to_json()],
            )
            path, pins = draws.training_view(
                root / "inputs", policy, views_root=root / "views"
            )
            self.assertFalse(
                validate_pairs_file(path, system_prompt=policy.system_prompt)
            )
            self.assertEqual(pins["style_system"], policy.system_prompt)
            self.assertEqual(pins["style_system_sha"], system_sha(policy.system_prompt))
            backend = backends.opts_from_manifest(
                {"data": pins}, default_system=policy.system_prompt
            )
            self.assertEqual(backend.system(), policy.system_prompt)
            self.assertNotIn("White", backend.system())

    def test_serving_refuses_prompt_mismatch_or_untrained_manifest(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            serving_options({"data": {"style_system": "old prompt"}}, "new prompt")
        with self.assertRaisesRegex(ValueError, "trained sampler"):
            serving_options({"data": {"style_system": "same prompt"}}, "same prompt")

    def test_serving_uses_checkpoint_and_exact_prompt(self):
        manifest = {
            "data": {"style_system": "Trained voice."},
            "result": {"checkpoints": {"sampler_path": "tinker://fixture/sampler"}},
            "hyperparameters": {
                "base_model": "Qwen/Qwen3.5-9B",
                "renderer_name": "qwen3_5_disable_thinking",
            },
        }
        backend = serving_options(manifest, "Trained voice.")
        self.assertEqual(backend.system(), "Trained voice.")
        self.assertEqual(backend.sampler_path, "tinker://fixture/sampler")
        self.assertEqual(backend.base_model, "Qwen/Qwen3.5-9B")


if __name__ == "__main__":
    unittest.main()
