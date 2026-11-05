"""Validate local synthetic drafts and export Stylebot-format pairs."""

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path

from stylebot.ai_core import style_system
from stylebot.pairs import system_sha, validate_pairs_file

PROMPT = style_system("E. B. White")
CAPITALIZED = re.compile(
    r"(?<![\w’'-])(?:[A-Z][a-z]+(?:[’'][A-Za-z]+)?|[A-Z]{2,})(?![\w’'-])"
)
ORDINARY = set(
    "A An The In On At As With For From Of If When While After Before By But And Or So Then That This Those These Here There One Two First Finally Since Where Now We I Our He She His Her Its They Their It Mr Mrs Ms Dr Sir To New".split()
)
WORDS = re.compile(r"\w+(?:[’'-]\w+)*")
NUMERIC = re.compile(r"(?<![\w])\d+(?:[,\.]\d+)*(?:%|st|nd|rd|th)?", re.I)
QUOTES = re.compile(r'“([^”\n]{1,550})”|"([^"\n]{1,550})"')
PREAMBLE = re.compile(
    r"^(?:#{1,6}\s|```|(?:rewritten passage|here(?:’|\x27)s|here is|as requested)\b.*[:\n])",
    re.I,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text):
    return " ".join(text.casefold().replace("’", "'").split())


def significant_names(text):
    names = set()
    for m in CAPITALIZED.finditer(text):
        term = m[0]
        if term in ORDINARY or (term.endswith("s") and term[:-1] in ORDINARY):
            continue
        prefix = text[: m.start()].rstrip(" \n\t“‘\"'()")
        follows_sentence_end = not prefix or prefix[-1] in ".?!:"
        if not follows_sentence_end or term.isupper():
            names.add(normalize(term).removesuffix("'s"))
    return names


def anchors(text):
    return set(n.replace(",", "").lower() for n in NUMERIC.findall(text))


def pronouns(text):
    # Skip quoted speech so it does not establish the narrator's viewpoint.
    tokens = set(WORDS.findall(normalize(QUOTES.sub("", text))))
    if tokens & {"i", "me", "my", "myself", "i'm", "i've", "i'd"}:
        return "singular"
    if tokens & {"we", "us", "our", "ourselves", "we're", "we've"}:
        return "plural"
    return ""


def check(target, draft):
    target_words = WORDS.findall(normalize(target))
    input_words = WORDS.findall(normalize(draft))
    ratio = len(input_words) / max(1, len(target_words))
    similarity = SequenceMatcher(
        None, target_words, input_words, autojunk=False
    ).ratio()
    flags, details = [], {}
    if (
        not draft
        or PREAMBLE.search(draft)
        or re.search(r"<\|[^>]+\|>|^[-*]\s", draft, re.M)
    ):
        flags.append("empty_or_non_prose")
    if not re.search(r"[.!?…][”’\"\')\]]*\s*$", draft):
        flags.append("unfinished")
    if not 0.72 <= ratio <= 1.60:
        flags.append("length_ratio")
    if not 0.16 <= similarity <= 0.84:
        flags.append("copy_similarity")
    if len(re.split(r"\n\s*\n", target)) != len(re.split(r"\n\s*\n", draft)):
        flags.append("paragraph_breaks")
    if anchors(draft) != anchors(target):
        flags.append("number_change")
        details["missing_numbers"] = sorted(anchors(target) - anchors(draft))
        details["added_numbers"] = sorted(anchors(draft) - anchors(target))
    lost_quotes = []
    draft_norm = normalize(draft)
    for match in QUOTES.finditer(target):
        quote = (match[1] or match[2]).strip()
        key = normalize(quote).rstrip(".,!?;:")
        if len(key.split()) >= 3 and key not in draft_norm:
            lost_quotes.append(quote)
    if lost_quotes:
        flags.append("quotation_change")
        details["changed_quote_count"] = len(lost_quotes)
    draft_word_set = {w.removesuffix("'s") for w in input_words}
    missing_names = significant_names(target) - draft_word_set
    # Content may have names split by a dash or punctuation in one version.
    missing_names = {
        w
        for w in missing_names
        if not re.search(r"\b" + re.escape(w) + r"\b", draft_norm)
    }
    if missing_names:
        flags.append("capitalized_term_missing")
        details["missing_terms"] = sorted(missing_names)
    if pronouns(target) != pronouns(draft):
        flags.append("viewpoint_change")
    return {
        "flags": flags,
        "similarity": round(similarity, 4),
        "length_ratio": round(ratio, 4),
        **details,
    }


def make_pair(passage, draft, check_result, stamp):
    source = passage["source"]
    target = passage["target"]
    key = sha256(
        f"{source}\0{passage['id']}\0{passage['target_sha256']}\0{draft['model']}".encode()
    ).hexdigest()[:16]
    meta = {
        "source": source,
        "captured_at": stamp,
        "capture_id": key,
        "chunk_index": passage["chunk_index"],
        "chunk_total": passage["chunk_total"],
        "before_chars": len(draft["text"]),
        "after_chars": len(target),
        "transform_sim": check_result["similarity"],
        "synthetic": True,
        "generator": draft["model"],
        "slop_strategy": draft["strategy"],
        "synth_key": key,
        "system_sha": system_sha(PROMPT),
        "tags": ["synthetic", "eb-white", "new-yorker"],
        "gen": {
            "model": draft["model"],
            "provider": draft["provider"],
            "prompt_label": draft["strategy"],
            "prompt_version": draft["prompt_version"],
            "temperature": draft.get("temperature"),
            "gen_seconds": draft.get("seconds"),
            "cost": 0 if draft["provider"].endswith("on-device") else None,
        },
        "source_title": passage["title"],
        "source_date": passage["date"],
        "source_section": passage["section"],
        "source_lines": passage["source_lines"],
        "target_sha256": passage["target_sha256"],
        "passage_id": passage["id"],
        "split": passage["split"],
        "quality_checks": check_result,
    }
    return {
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": draft["text"]},
            {"role": "assistant", "content": target},
        ],
        "meta": meta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    work_dir, output_dir = args.work_dir, args.output_dir
    stats = json.loads((work_dir / "source-stats.json").read_text())
    passage_list = read_jsonl(work_dir / "passages.jsonl")
    passages = {p["id"]: p for p in passage_list}
    generated = []
    for name in (
        "drafts.jsonl",
        "retry-output.jsonl",
        "qwen-output.jsonl",
        "manual-output.jsonl",
    ):
        path = work_dir / name
        if path.exists():
            for raw in read_jsonl(path):
                raw.setdefault("model", "apple-system-language-model")
                raw.setdefault("provider", "Apple Foundation Models on-device")
                raw.setdefault("prompt_version", 2)
                raw.setdefault("temperature", 0.65)
                if raw.get("text"):
                    raw["text"] = raw["text"].strip()
                generated.append(raw)
    review_file = work_dir / "manual-review.json"
    review = json.loads(review_file.read_text()) if review_file.exists() else {}
    outcomes = {}
    accepted = {}
    stamp = stats["prepared_at"]
    for raw in generated:
        if raw["id"] not in passages:
            continue
        passage = passages[raw["id"]]
        draft = raw.get("text")
        result = (
            check(passage["target"], draft)
            if draft
            else {"flags": ["generation_error"]}
        )
        audit = review.get(raw["id"])
        if audit and raw["model"] == audit["model"]:
            result["review"] = audit["reason"]
            if audit.get("reject"):
                result["flags"].append("spot_check_rejection")
        if not result["flags"]:
            accepted[passage["id"]] = make_pair(passage, raw, result, stamp)
        outcomes[passage["id"]] = result
    all_pieces = read_jsonl(work_dir / "pieces.jsonl")
    stats["prose_exclusions"] = {
        p["source"]: p["excluded_reason"]
        for p in all_pieces
        if p.get("excluded_reason")
    }
    stats["target_passages"] = Counter(p["split"] for p in passage_list)
    stats["target_words"] = {
        split: sum(p["words"] for p in passage_list if p["split"] == split)
        for split in ("train", "eval")
    }
    stats["attempted_passages"] = Counter(passages[k]["split"] for k in outcomes)
    stats["generation_calls"] = len(generated)
    stats["accepted_pairs"] = Counter(passages[k]["split"] for k in accepted)
    stats["accepted_sources"] = {
        split: len(
            {passages[k]["source"] for k in accepted if passages[k]["split"] == split}
        )
        for split in ("train", "eval")
    }
    stats["accepted_target_words"] = {
        split: sum(
            passages[k]["words"] for k in accepted if passages[k]["split"] == split
        )
        for split in ("train", "eval")
    }
    stats["rejection_reasons"] = Counter(
        flag
        for k, result in outcomes.items()
        if k not in accepted
        for flag in result["flags"]
    )
    stats["accepted_by_generator"] = Counter(
        row["meta"]["generator"] for row in accepted.values()
    )
    stats["accepted_training_sections"] = Counter(
        passages[k]["section"] for k in accepted if passages[k]["split"] == "train"
    )
    stats["manually_reviewed_passages"] = len(review)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.stats_only:
        return
    if not any(p["meta"]["split"] == "train" for p in accepted.values()):
        parser.error(
            "No accepted training pairs. Generate drafts and review the quality report."
        )
    paths = {
        "train": output_dir / "pairs.jsonl",
        "eval": output_dir / "evaluation" / "pairs.jsonl",
    }
    for split, path in paths.items():
        pairs = [
            accepted[p["id"]]
            for p in passage_list
            if p["split"] == split and p["id"] in accepted
        ]
        write_jsonl(path, pairs)
        errors = validate_pairs_file(path, system_prompt=PROMPT)
        assert not errors, errors
    write_jsonl(
        output_dir / "pieces.jsonl",
        [
            {k: v for k, v in p.items() if k not in ("original", "paragraphs")}
            for p in all_pieces
        ],
    )
    write_jsonl(
        output_dir / "passages.jsonl",
        [
            p
            | {
                "pair_status": (
                    "accepted"
                    if p["id"] in accepted
                    else "rejected"
                    if p["id"] in outcomes
                    else "pending"
                ),
                "rejection_reasons": (
                    outcomes[p["id"]]["flags"]
                    if p["id"] in outcomes and p["id"] not in accepted
                    else []
                ),
            }
            for p in passage_list
        ],
    )
    train = read_jsonl(paths["train"])
    evaluation = read_jsonl(paths["eval"])
    assert not {r["meta"]["source"] for r in train} & {
        r["meta"]["source"] for r in evaluation
    }
    assert not {r["meta"]["target_sha256"] for r in train} & {
        r["meta"]["target_sha256"] for r in evaluation
    }
    assert all(
        r["messages"][2]["content"] == passages[r["meta"]["passage_id"]]["target"]
        for r in train + evaluation
    )
    assert all(
        r["messages"][1]["content"] != r["messages"][2]["content"]
        for r in train + evaluation
    )
    (output_dir / "policy.json").write_text(
        json.dumps({"system_prompt": PROMPT}, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "dataset-summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
