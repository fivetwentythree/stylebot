"""Prepare training-only context and record fidelity decisions for exact pairs."""

import argparse
import json
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_sha(text):
    return sha256(text.encode("utf-8")).hexdigest()


def index_unique(path):
    result = {}
    for row in read_jsonl(path):
        if row["id"] in result:
            raise ValueError(f"Duplicate passage ID {row['id']} in {path}")
        result[row["id"]] = row
    return result


def with_neighbors(passage, piece, *, words=85):
    body = " ".join(" ".join(piece["paragraphs"]).split())
    target = " ".join(passage["target"].split())
    start = body.find(target)
    if start < 0 or body.find(target, start + 1) >= 0:
        raise ValueError(f"Cannot locate a unique source span for {passage['id']}")
    end = start + len(target)
    if (start > 0 and body[start - 1] != " ") or (end < len(body) and body[end] != " "):
        raise ValueError(f"Invalid source boundaries for {passage['id']}")
    return {
        **passage,
        "context_before": " ".join(body[:start].split()[-words:]),
        "context_after": " ".join(body[end:].split()[:words]),
    }


def select_passages(work_dir, dataset_dir, limit):
    passages = read_jsonl(work_dir / "passages.jsonl")
    pieces = {p["source"]: p for p in read_jsonl(work_dir / "pieces.jsonl")}
    accepted = read_jsonl(dataset_dir / "pairs.jsonl")
    held = read_jsonl(dataset_dir / "evaluation" / "pairs.jsonl")
    held_sources = {p["meta"]["source"] for p in held}
    used = {p["meta"]["passage_id"] for p in accepted}
    counts = Counter(p["meta"]["source"] for p in accepted)
    status = {p["id"]: p for p in read_jsonl(dataset_dir / "passages.jsonl")}
    candidates = defaultdict(list)
    for passage in passages:
        source = passage["source"]
        prior = status.get(passage["id"], {})
        if (
            passage["split"] == "train"
            and pieces[source]["split"] == "train"
            and source not in held_sources
            and passage["id"] not in used
            and "uncertain_source_transcription"
            not in prior.get("rejection_reasons", [])
        ):
            candidates[source].append(passage)

    per_book = defaultdict(list)
    for source in sorted(candidates, key=lambda source: (counts[source], source)):
        group = sorted(
            candidates[source],
            key=lambda p: (
                status.get(p["id"], {}).get("pair_status") != "pending",
                abs(p["words"] - 170),
                p["chunk_index"],
            ),
        )
        per_book[source.split("/", 1)[0]].append(group)

    queues = [
        [p for group in zip_longest(*groups) for p in group if p is not None]
        for groups in per_book.values()
    ]
    selected = [p for group in zip_longest(*queues) for p in group if p is not None]
    return [with_neighbors(p, pieces[p["source"]]) for p in selected[:limit]]


def record_decisions(passages_path, drafts_path, decisions_path):
    passages = index_unique(passages_path)
    drafts = index_unique(drafts_path)
    for decision in index_unique(decisions_path).values():
        passage = passages[decision["id"]]
        draft = drafts[decision["id"]]
        draft_text = (draft.get("text") or "").strip()
        if (
            passage["split"] != "train"
            or text_sha(passage["target"]) != passage["target_sha256"]
        ):
            raise ValueError(f"Invalid training target for {passage['id']}")
        if decision["target_sha256"] != passage["target_sha256"] or decision[
            "draft_sha256"
        ] != text_sha(draft_text):
            raise ValueError(f"Stale fidelity decision for {passage['id']}")
        if (
            decision["verdict"] not in {"pass", "reject"}
            or not decision["reviewer"].strip()
            or not decision["note"].strip()
            or (decision["verdict"] == "pass" and not draft_text)
        ):
            raise ValueError(f"Incomplete fidelity decision for {passage['id']}")
        yield {
            **draft,
            "text": draft_text,
            "target_sha256": passage["target_sha256"],
            "fidelity": {
                "verdict": decision["verdict"],
                "reviewer": decision["reviewer"],
                "note": decision["note"],
                "draft_sha256": decision["draft_sha256"],
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare")
    prepare.add_argument("work_dir", type=Path)
    prepare.add_argument("dataset_dir", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--limit", type=int, default=50)
    record = modes.add_parser("record")
    record.add_argument("passages", type=Path)
    record.add_argument("drafts", type=Path)
    record.add_argument("decisions", type=Path)
    record.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        rows = select_passages(args.work_dir, args.dataset_dir, args.limit)
    else:
        rows = list(record_decisions(args.passages, args.drafts, args.decisions))
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
