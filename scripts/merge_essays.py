"""Add Essays of E. B. White to a prepared corpus, keeping the older split fixed."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from prepare import clean_piece, pack, split_long_paragraph, write_jsonl

TITLES = (
    "Good-bye to Forty-eighth Street",
    "Home-Coming",
    "A Report in Spring",
    "Death of a Pig",
    "The Eye of Edna",
    "Coon Tree",
    "A Report in January",
    "The Winter of the Great Snows",
    "Riposte",
    "The Geese",
    "Letter from the East",
    "Bedfellows",
    "Sootfall and Fallout",
    "Unity",
    "The World of Tomorrow",
    "Here Is New York",
    "On a Florida Key",
    "The Ring of Time",
    "What Do Our Hearts Treasure?",
    "Afternoon of an American Boy",
    "Farewell, My Lovely!",
    "The Years of Wonder",
    "Once More to the Lake",
    "The Sea and the Wind That Blows",
    "The Railroad",
    "The St. Nicholas League",
    "A Slight Sound at Evening",
    "Some Remarks on Humor",
    "Don Marquis",
    "Will Strunk",
    "Mr. Forbush’s Friends",
)
SECTIONS = (
    "THE FARM",
    "THE PLANET",
    "THE CITY",
    "FLORIDA",
    "MEMORIES",
    "DIVERSIONS AND OBSESSIONS",
    "BOOKS, MEN, AND WRITING",
)
HOLDOUT = {
    "The Geese",
    "The World of Tomorrow",
    "The Ring of Time",
    "Once More to the Lake",
    "A Slight Sound at Evening",
}
DATELINE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Spring|Summer|Autumn|Winter)\b.*\d{4}",
    re.I,
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def word_shingles(text, width):
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return set(zip(*(words[i:] for i in range(width))))


def extract_essays(raw, first_id):
    lines = raw.decode("utf-8-sig").splitlines()
    headings = [
        (i, line.strip().rstrip("*"))
        for i, line in enumerate(lines)
        if line.strip().rstrip("*") in TITLES
    ]
    if tuple(title for _, title in headings) != TITLES:
        raise ValueError(
            "Expected the 31 essay headings from the supplied text export, in order."
        )
    sections = {
        i: line.strip() for i, line in enumerate(lines) if line.strip() in SECTIONS
    }
    structural = set(sections) | {
        i for i, line in enumerate(lines) if re.fullmatch(r"[IVX]{1,5}", line.strip())
    }
    pieces = []
    for offset, (heading, title) in enumerate(headings):
        end = headings[offset + 1][0] if offset + 1 < len(headings) else len(lines)
        end = min([end, *(i for i in structural if heading < i < end)])
        body = [
            (i + 1, lines[i].strip())
            for i in range(heading + 1, end)
            if lines[i].strip()
        ]
        date = None
        if body[0][1].isupper() and DATELINE.search(body[0][1]):
            date = body.pop(0)[1]
        body = [
            (line, text) for line, text in body if text not in {"* * *", "E. B. White"}
        ]
        if title == "Farewell, My Lovely!" and body[0][1].startswith("("):
            body.pop(0)  # subtitle
        if title == "The Railroad":
            body = body[
                next(
                    i
                    for i, (_, text) in enumerate(body)
                    if text.startswith("Henry Thoreau, who wrote those lines")
                ) :
            ]
        line_numbers, paragraphs = zip(*body)
        original = "\n\n".join(paragraphs)
        section_index = max(i for i in sections if i < heading)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        piece = dict(
            source=f"essays/{first_id + offset:03d}-{slug}.md",
            title=title,
            date=date,
            section=sections[section_index].title(),
            section_index=list(sections).index(section_index) + 1,
            source_lines=[line_numbers[0], line_numbers[-1]],
            words=len(original.split()),
            original=original,
            paragraphs=clean_piece(paragraphs),
            split="eval" if title in HOLDOUT else "train",
        )
        if title == "The St. Nicholas League":
            piece["excluded_reason"] = "contains extensive quoted work by other writers"
        pieces.append(piece)
    return pieces


class OverlapIndex:
    """Index exact 24-word spans and 12-word fingerprints of near-reprints."""

    def __init__(self, passages):
        self.exact = {}
        self.near = defaultdict(set)
        self.sizes = {}
        for passage in passages:
            self.add(passage["id"], passage["target"])

    def add(self, key, text):
        for fingerprint in word_shingles(text, 24):
            self.exact.setdefault(fingerprint, key)
        fingerprints = word_shingles(text, 12)
        self.sizes[key] = len(fingerprints)
        for fingerprint in fingerprints:
            self.near[fingerprint].add(key)

    def matches(self, text):
        exact = {self.exact[s] for s in word_shingles(text, 24) if s in self.exact}
        fingerprints = word_shingles(text, 12)
        counts = Counter(
            key
            for fingerprint in fingerprints
            for key in self.near.get(fingerprint, ())
        )
        return sorted(
            exact
            | {
                key
                for key, count in counts.items()
                if count >= 12
                and count / min(len(fingerprints), self.sizes[key]) >= 0.55
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", type=Path)
    parser.add_argument("previous_work_dir", type=Path)
    parser.add_argument("combined_work_dir", type=Path)
    args = parser.parse_args()
    old_dir, out = args.previous_work_dir, args.combined_work_dir
    if old_dir.resolve() == out.resolve():
        parser.error(
            "Use a separate combined directory to preserve the original split and IDs."
        )
    old_pieces = read_jsonl(old_dir / "pieces.jsonl")
    old_passages = read_jsonl(old_dir / "passages.jsonl")
    first_id = max(int(p["source"].split("/")[1].split("-")[0]) for p in old_pieces) + 1
    raw = args.book.read_bytes()
    new_pieces = extract_essays(raw, first_id)
    index = OverlapIndex(old_passages)
    overlaps = []
    passages = old_passages.copy()
    for piece in sorted(new_pieces, key=lambda p: (p["split"] != "eval", p["source"])):
        if piece.get("excluded_reason"):
            continue
        units = [
            unit for para in piece["paragraphs"] for unit in split_long_paragraph(para)
        ]
        accepted = []
        for chunk_no, chunk in enumerate(pack(units)):
            text = "\n\n".join(chunk)
            words = len(text.split())
            if not 55 <= words <= 410:
                continue
            key = f"{piece['source'].split('/')[1][:3]}-{chunk_no + 1:02d}"
            matches = index.matches(text)
            if matches:
                overlaps.append(
                    {"source": piece["source"], "passage_id": key, "matches": matches}
                )
                continue
            record = {
                k: piece[k]
                for k in ("source", "title", "date", "section", "split", "source_lines")
            }
            record.update(
                id=key,
                target=text,
                words=words,
                target_sha256=sha256(text.encode()).hexdigest(),
            )
            accepted.append(record)
            index.add(key, text)
        for chunk_index, passage in enumerate(accepted):
            passage.update(chunk_index=chunk_index, chunk_total=len(accepted))
            passages.append(passage)
    pieces = old_pieces + new_pieces
    stats = json.loads((old_dir / "source-stats.json").read_text(encoding="utf-8"))
    stats["sources"] = [
        {"filename": stats.pop("source_file"), "sha256": stats.pop("source_sha256")},
        {"filename": args.book.name, "sha256": sha256(raw).hexdigest()},
    ]
    stats["prepared_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats["pieces"] = Counter(p["split"] for p in pieces)
    stats["piece_words"] = {
        split: sum(p["words"] for p in pieces if p["split"] == split)
        for split in ("train", "eval")
    }
    stats["target_passages"] = Counter(p["split"] for p in passages)
    stats["target_words"] = {
        split: sum(p["words"] for p in passages if p["split"] == split)
        for split in ("train", "eval")
    }
    stats["normalized_pieces"] = sum(
        p["original"] != "\n\n".join(p["paragraphs"]) for p in pieces
    )
    stats["duplicate_passages_excluded"] = len(overlaps)
    assert len({p["id"] for p in passages}) == len(passages)
    assert len({p["target_sha256"] for p in passages}) == len(passages)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "pieces.jsonl", pieces)
    write_jsonl(out / "passages.jsonl", passages)
    write_jsonl(out / "overlaps.jsonl", overlaps)
    (out / "source-stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
