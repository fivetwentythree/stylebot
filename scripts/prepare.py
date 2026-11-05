"""Extract and split the supplied E. B. White text export."""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}")

# Small capitals in the export are typography. Retain names and acronyms
# in opening phrases that need more than ordinary sentence casing.
OPENINGS = {
    "A DANE TOLD US": "A Dane told us",
    "THE Nation HAS PUT": "The Nation has put",
    "TO THE EDITOR OF": "To the Editor of",
    "GEORGE SELDES,": "George Seldes,",
    "ADLAI STEVENSON*": "Adlai Stevenson*",
    "MAY 6TH IS THE SADDEST DAY": "May 6th is the saddest day",
    "THE THOREAU SOCIETY": "The Thoreau Society",
    "WHEN SENATOR MCCARTHY*": "When Senator McCarthy*",
    "PRESIDENT TRUMAN SAYS": "President Truman says",
    "DR. SOCKMAN,": "Dr. Sockman,",
    "THE GERMANS,": "The Germans,",
    "MR. McNEIL* AND MR. VISHINSKY": "Mr. McNeil* and Mr. Vishinsky",
    "OUT IN SUMMIT, NEW JERSEY,": "Out in Summit, New Jersey,",
    "YESTERDAY IN THE GRAYBAR BUILDING I": "Yesterday in the Graybar Building I",
    "THE NEW REPTILE HALL": "The new Reptile Hall",
    "THE BOARD OF EDUCATION": "The Board of Education",
    "BETWEEN BERGEN EVANS": "Between Bergen Evans",
    "ON TV THE OTHER MORNING": "On TV the other morning",
    "EVERYONE HAS ONE CHRISTMAS": "Everyone has one Christmas",
    "SHOPPING IN WOOLWORTH’S,": "Shopping in Woolworth’s,",
    "WE HAD A SCROOGE": "We had a Scrooge",
    "AT CHRISTMAS ONE SHOULD": "At Christmas one should",
    "THE INQUIRING PHOTOGRAPHER": "The Inquiring Photographer",
    "THE LONELY MB. LINDBEBGH": "The lonely Mr. Lindbergh",
    "ROSS* DIED IN BOSTON,": "Ross* died in Boston,",
}

# Fix explicit extraction errors. Leave ambiguous period spellings intact.
OCR_FIXES = {
    "WALTE R TITHRIDGE": "WALTER TITHRIDGE",
    "hap-pens": "happens",
    "in-telligence": "intelligence",
    "de-spite": "despite",
    "Levia-than": "Leviathan",
    "Long Is-land": "Long Island",
    "“Waiden”": "“Walden”",
    "I’ironie": "l’ironie",
    "millions of Uves": "millions of lives",
    "world’send article": "world’s-end article",
    "at the takeofl": "at the takeoff",
}
OCR_WRAPS = set(
    "how-ever in-tense some-thing every-body be-hind un-likely back-ground fore-head en-danger some-what un-spoiled de-pressed de-void re-ally Some-times any-thing psycho-logical child-hood him-self some-body snap-ping be-loved an-other some-times for-gotten after-noon any-body be-cause over-powering un-kempt relation-ship s-end al-most un-equivocally al-ways in-creased contem-plates Eng-land com-plied be-wails bur-docks re-marks spar-rows".split()
)
WRAP_PATTERN = re.compile(r"\b(?:" + "|".join(sorted(OCR_WRAPS)) + r")\b")


def clean_opening(first):
    for raw, clean in OPENINGS.items():
        if first.startswith(raw):
            return clean + first[len(raw) :]
    tokens = list(re.finditer(r"\S+", first))
    run = []
    for token in tokens:
        if not token.group().isupper():
            break
        run.append(token)
    if not run:
        return first
    end = run[-1].end()
    prefix = first[:end].lower()
    prefix = re.sub(r"\bi(?=\b)", "I", prefix)
    prefix = re.sub(r"[a-z]", lambda m: m[0].upper(), prefix, count=1)
    return prefix + first[end:]


def clean_piece(paras):
    normalized = [clean_opening(paras[0]), *paras[1:]]
    for idx, para in enumerate(normalized):
        for raw, clean in OCR_FIXES.items():
            para = para.replace(raw, clean)
        normalized[idx] = WRAP_PATTERN.sub(lambda m: m[0].replace("-", ""), para)
    return normalized


def section_and_article_bounds(lines):
    sections = {}
    for idx, line in enumerate(lines):
        if line.strip().isdigit():
            section = int(line.strip())
            assert section == len(sections) + 1
            name_line = next(j for j in range(idx + 1, len(lines)) if lines[j].strip())
            sections[idx] = (section, lines[name_line].strip())
    headers = []
    for idx, line in enumerate(lines):
        if not DATE.fullmatch(line.strip()):
            continue
        head = next(j for j in range(idx - 1, -1, -1) if lines[j].strip())
        if lines[head].strip().startswith("("):
            head = next(j for j in range(head - 1, -1, -1) if lines[j].strip())
        title = " ".join(x.strip() for x in lines[head:idx] if x.strip())
        assert title.isupper(), (head + 1, title)
        section, section_title = sections[max(j for j in sections if j < head)]
        headers.append((head, idx, title, section, section_title))
    return sections, headers


def split_long_paragraph(para):
    # Avoid abbreviations and initials when finding sentences in long paragraphs.
    boundaries = []
    for m in re.finditer(r"[.!?][”’\"\)]*\s+(?=[“‘\"\(]*[A-Z])", para):
        before = para[: m.start()]
        if re.search(
            r"\b(?:[A-Z]|Mr|Mrs|Ms|Dr|Gen|Gov|Sen|St|Jr|Sr|Prof|Rev|etc|vs|U\.S)\Z",
            before,
        ):
            continue
        boundaries.append(m.end())
    cuts = [0, *boundaries, len(para)]
    if len(para.split()) <= 265 or len(cuts) <= 2:
        return [para]
    sentences = [para[a:b].strip() for a, b in zip(cuts, cuts[1:])]
    return [" ".join(part) for part in pack(sentences, max_words=235, min_words=125)]


def pack(units, *, max_words=265, min_words=145):
    groups, current, words = [], [], 0
    for unit in units:
        size = len(unit.split())
        if current and words + size > max_words:
            groups.append(current)
            current, words = [], 0
        current.append(unit)
        words += size
        if words >= min_words:
            groups.append(current)
            current, words = [], 0
    if current:
        if groups and sum(len(x.split()) for x in groups[-1] + current) <= max_words:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def choose_holdout(pieces):
    # Spread whole-piece evaluation across subjects and hold out close to 13% of words.
    per_section = {}
    for piece in pieces:
        per_section.setdefault(piece["section_index"], []).append(piece)
    held = set()
    for section, group in per_section.items():
        pool = [p for p in group if 125 <= p["words"] <= 700] or group
        center = sum(p["words"] for p in group) / len(group)
        choice = sorted(pool, key=lambda p: (abs(p["words"] - center), p["source"]))[0]
        held.add(choice["source"])
    for piece in sorted(pieces, key=lambda p: p["source"]):
        if sum(p["words"] for p in pieces if p["source"] in held) >= 0.13 * sum(
            p["words"] for p in pieces
        ):
            break
        if 220 <= piece["words"] <= 650:
            held.add(piece["source"])
    return held


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source_path, output_path = args.book, args.output_dir
    raw = source_path.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    sections, headers = section_and_article_bounds(lines)
    if not headers or not sections:
        raise ValueError(
            "No dated pieces found. This extractor expects the New Yorker text export."
        )
    pieces = []
    for order, (head, dated, title, sec, section_title) in enumerate(headers, 1):
        end = headers[order][0] if order < len(headers) else len(lines)
        end = min([end, *(i for i in sections if dated < i < end)])
        paragraph_lines = [
            (i + 1, lines[i].strip()) for i in range(dated + 1, end) if lines[i].strip()
        ]
        assert paragraph_lines
        line_numbers, paras = zip(*paragraph_lines)
        original = "\n\n".join(paras)
        clean = clean_piece(paras)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]
        source_id = f"new-yorker/{order:03d}-{slug}.md"
        pieces.append(
            dict(
                source=source_id,
                title=title,
                date=lines[dated].strip(),
                section=section_title,
                section_index=sec,
                source_lines=[line_numbers[0], line_numbers[-1]],
                words=len(original.split()),
                original=original,
                paragraphs=clean,
            )
        )

    held = choose_holdout(pieces)
    rows = []
    seen = set()
    for piece in pieces:
        group = "eval" if piece["source"] in held else "train"
        piece["split"] = group
        if piece["title"] == "VERMIN":
            piece["excluded_reason"] = "verse"
            continue
        candidates = [
            p for para in piece["paragraphs"] for p in split_long_paragraph(para)
        ]
        # Keep very short full pieces when they still give a meaningful prose example.
        blocks = ["\n\n".join(block) for block in pack(candidates)]
        blocks = [block for block in blocks if 55 <= len(block.split()) <= 410]
        for index, target in enumerate(blocks):
            fingerprint = hashlib.sha256(
                " ".join(target.casefold().split()).encode()
            ).hexdigest()
            assert fingerprint not in seen, (piece["source"], index)
            seen.add(fingerprint)
            record = {
                k: piece[k]
                for k in ("source", "title", "date", "section", "split", "source_lines")
            }
            record.update(
                id=f"{piece['source'].split('/')[1][:3]}-{index + 1:02d}",
                target=target,
                words=len(target.split()),
                chunk_index=index,
                chunk_total=len(blocks),
                target_sha256=hashlib.sha256(target.encode()).hexdigest(),
            )
            rows.append(record)

    metadata = {
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": source_path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "pieces": Counter(p["split"] for p in pieces),
        "piece_words": {
            split: sum(p["words"] for p in pieces if p["split"] == split)
            for split in ("train", "eval")
        },
        "target_passages": Counter(r["split"] for r in rows),
        "target_words": {
            split: sum(r["words"] for r in rows if r["split"] == split)
            for split in ("train", "eval")
        },
        "normalized_pieces": sum(
            p["original"] != "\n\n".join(p["paragraphs"]) for p in pieces
        ),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path / "pieces.jsonl", pieces)
    write_jsonl(output_path / "passages.jsonl", rows)
    (output_path / "source-stats.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
