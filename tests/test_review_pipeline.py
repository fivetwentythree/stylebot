"""Regression checks for review integrity, split isolation and prompt propagation."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from package import apply_fidelity_review, pronouns
from audit_pairs import apply_legacy_audit, evaluation_eligibility, validate_audit
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
            self.passage
            | {
                "id": "excluded-target",
                "source": "book/quarantined.md",
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
        (work / "target-exclusions.json").write_text(
            json.dumps({"excluded-target": "Target needs transcription verification."})
        )
        selected = select_passages(work, data, 10)
        self.assertEqual([row["source"] for row in selected], ["book/safe.md"])


class LegacyAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        repo = Path(__file__).resolve().parents[1]
        self.row = json.loads((repo / "examples" / "pairs.jsonl").read_text())
        self.row["meta"] |= {"passage_id": "fixture-01", "split": "train"}
        meta = self.row["meta"]
        self.note = {
            "id": "fixture-01",
            "split": "train",
            "source": meta["source"],
            "generator": meta["generator"],
            "capture_id": meta["capture_id"],
            "target_sha256": text_sha(self.row["messages"][2]["content"]),
            "draft_sha256": text_sha(self.row["messages"][1]["content"]),
            "verdict": "pass",
            "reviewer": "test reviewer",
            "note": "Fixture reviewed.",
        }
        self.passage = {
            "id": "fixture-01",
            "source": meta["source"],
            "split": "train",
            "target_sha256": self.note["target_sha256"],
        }
        self.raw = {
            "id": "fixture-01",
            "model": meta["generator"],
            "text": self.row["messages"][1]["content"],
        }

    def validated(self, notes):
        snapshot, decisions = self.root / "snapshot", self.root / "decisions"
        write_jsonl(snapshot, [self.row])
        write_jsonl(decisions, notes)
        return list(validate_audit(snapshot, decisions))

    def test_audit_requires_complete_snapshot_coverage(self):
        self.assertEqual(self.validated([self.note]), [self.note])
        with self.assertRaisesRegex(ValueError, "every pair"):
            self.validated([])

    def test_volume_roman_numeral_does_not_hide_lost_narrator(self):
        self.assertEqual(pronouns("The initial entry in Volume I is a bird."), "")
        self.assertEqual(
            pronouns("Volume I starts with a bird. I read it."), "singular"
        )
        self.assertEqual(pronouns("This is the volume I read."), "singular")

    def test_audit_rejects_changed_content_or_provenance(self):
        for field in ("target_sha256", "draft_sha256", "source", "generator", "split"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Stale or misattributed"):
                    self.validated([self.note | {field: "changed"}])

    def test_legacy_rejection_and_unreviewed_alternative_cannot_pass(self):
        result = {"flags": []}
        apply_legacy_audit(
            self.passage,
            self.raw,
            result,
            {"fixture-01": self.note | {"verdict": "reject"}},
        )
        self.assertIn("legacy_fidelity_not_approved", result["flags"])
        result = {"flags": []}
        apply_legacy_audit(
            self.passage,
            self.raw | {"text": "A different unreviewed draft."},
            result,
            {"fixture-01": self.note},
        )
        self.assertIn("unreviewed_legacy_variant", result["flags"])

    def test_unchanged_approved_legacy_pair_passes(self):
        result = {"flags": []}
        apply_legacy_audit(self.passage, self.raw, result, {"fixture-01": self.note})
        self.assertEqual(result["flags"], [])
        self.assertEqual(result["legacy_review"], self.note)

    def make_evaluation(self):
        self.row["meta"]["split"] = self.passage["split"] = self.note["split"] = "eval"

    def test_failed_evaluation_is_preserved_but_not_scored(self):
        self.make_evaluation()
        self.note["verdict"] = "reject"
        before = copy.deepcopy(self.row)
        result = {"flags": []}
        apply_legacy_audit(self.passage, self.raw, result, {"fixture-01": self.note})
        report, eligible = evaluation_eligibility(
            [self.row],
            {"fixture-01": self.passage},
            {"fixture-01": self.note},
        )
        self.assertEqual(self.row, before)
        self.assertEqual(result["flags"], [])
        self.assertEqual(eligible, [])
        self.assertTrue(report[0]["fingerprints_match"])
        self.assertFalse(report[0]["scoring_eligible"])

    def test_scoring_requires_exact_approved_evaluation_messages(self):
        self.make_evaluation()
        _, eligible = evaluation_eligibility(
            [self.row],
            {"fixture-01": self.passage},
            {"fixture-01": self.note},
        )
        self.assertEqual(eligible, [self.row])
        for message_index in (1, 2):
            with self.subTest(message_index=message_index):
                changed = copy.deepcopy(self.row)
                changed["messages"][message_index]["content"] += " Changed."
                _, eligible = evaluation_eligibility(
                    [changed],
                    {"fixture-01": self.passage},
                    {"fixture-01": self.note},
                )
                self.assertEqual(eligible, [])


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
