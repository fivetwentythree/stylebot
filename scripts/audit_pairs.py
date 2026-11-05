"""Validate complete semantic reviews of an existing accepted-pair snapshot."""

import argparse
from pathlib import Path

from review_batch import index_unique, read_jsonl, text_sha, write_jsonl


def validate_audit(snapshot, decisions):
    pairs = {}
    for row in read_jsonl(snapshot):
        pid = row["meta"]["passage_id"]
        if pid in pairs:
            raise ValueError(f"Duplicate passage ID {pid} in snapshot")
        pairs[pid] = row
    notes = index_unique(decisions)
    if pairs.keys() != notes.keys():
        raise ValueError(
            "The audit must cover every pair in its snapshot exactly once."
        )
    for pid, row in pairs.items():
        note = notes[pid]
        meta = row["meta"]
        expected = {
            "id": pid,
            "split": meta["split"],
            "source": meta["source"],
            "generator": meta["generator"],
            "capture_id": meta["capture_id"],
            "target_sha256": text_sha(row["messages"][2]["content"]),
            "draft_sha256": text_sha(row["messages"][1]["content"].strip()),
        }
        if any(note.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Stale or misattributed audit for {pid}")
        if (
            note.get("verdict") not in {"pass", "reject"}
            or not note.get("reviewer", "").strip()
            or not note.get("note", "").strip()
        ):
            raise ValueError(f"Incomplete audit for {pid}")
        yield note


def audit_matches(note, passage, raw):
    return (
        note["id"] == passage["id"]
        and note["split"] == passage["split"]
        and note["source"] == passage["source"]
        and note["target_sha256"] == passage["target_sha256"]
        and note["generator"] == raw["model"]
        and note["draft_sha256"] == text_sha((raw.get("text") or "").strip())
    )


def apply_legacy_audit(passage, raw, result, audits):
    # Existing evaluation messages remain frozen. Their scoring eligibility is
    # exported separately after matching the same exact-pair fingerprints.
    if passage["split"] != "train" or raw.get("requires_review"):
        return
    note = audits.get(passage["id"])
    if note is None or not audit_matches(note, passage, raw):
        result["flags"].append("unreviewed_legacy_variant")
        return
    result["legacy_review"] = note
    if note["verdict"] != "pass":
        result["flags"].append("legacy_fidelity_not_approved")


def evaluation_eligibility(evaluation, passages, audits):
    reports, eligible = [], []
    for row in evaluation:
        meta = row["meta"]
        pid = meta["passage_id"]
        note = audits.get(pid)
        raw = {"model": meta["generator"], "text": row["messages"][1]["content"]}
        exact = bool(
            note
            and audit_matches(note, passages[pid], raw)
            and note["capture_id"] == meta["capture_id"]
            and note["target_sha256"] == text_sha(row["messages"][2]["content"])
        )
        approved = exact and note["verdict"] == "pass"
        reports.append(
            {
                "id": pid,
                "scoring_eligible": approved,
                "review": note,
                "fingerprints_match": exact,
            }
        )
        if approved:
            eligible.append(row)
    return reports, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    notes = list(validate_audit(args.snapshot, args.decisions))
    write_jsonl(args.output, notes)
    print(f"Validated {len(notes)} exact-pair reviews into {args.output}")


if __name__ == "__main__":
    main()
