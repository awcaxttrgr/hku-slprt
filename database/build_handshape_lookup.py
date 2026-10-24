import json
from pathlib import Path


DATABASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = DATABASE_DIR / "data.json"
OUTPUT_PATH = DATABASE_DIR / "data.lookup.json"
LOOKUP_SCHEMA_VERSION = 8
PRECOMPUTED_MATCH_LIMIT = 60


def parse_codepoint_pairs(notation):
    pairs = []
    for group in notation.replace(".", ",").split(","):
        values = [int(value) for value in group.split(";") if value]
        if len(values) % 2:
            values.append(0)
        for index in range(0, len(values), 2):
            pairs.append([values[index], values[index + 1]])
    return pairs


def match_rank(entry, right_codepoint, left_codepoint):
    best_rank = None
    for pair_index, (dominant, non_dominant) in enumerate(entry["pairs"]):
        rank = None
        if (
            right_codepoint is not None
            and left_codepoint is not None
            and dominant != 0
            and non_dominant != 0
            and (
                (
                    right_codepoint == dominant
                    and left_codepoint == non_dominant
                )
                or (
                    left_codepoint == dominant
                    and right_codepoint == non_dominant
                )
            )
        ):
            rank = 0
        elif dominant != 0 and dominant in (
            right_codepoint,
            left_codepoint,
        ):
            rank = 1
        elif non_dominant != 0 and non_dominant in (
            right_codepoint,
            left_codepoint,
        ):
            rank = 2

        if rank is not None:
            candidate_rank = (pair_index, rank)
            if best_rank is None or candidate_rank < best_rank:
                best_rank = candidate_rank
    return best_rank


def build_lookup():
    with SOURCE_PATH.open(encoding="utf-8") as source_file:
        source_entries = json.load(source_file)

    entries = []
    by_codepoint = {}
    video_names = set()
    missing_videos = 0
    for entry in source_entries:
        record = entry.get("HKU_HKSLLEX_ID") or {}
        if record.get("status") == "under_review":
            continue

        notation = (
            entry.get("cuhkcslds_HandshapeFont") or {}
        ).get("CodepointNotation")
        video = (entry.get("videoFile") or {}).get("nameGIF")
        if not notation or not video:
            continue
        video = Path(video).name
        if not (DATABASE_DIR / "webp" / video).is_file():
            missing_videos += 1
            continue
        if video in video_names:
            raise ValueError(f"Duplicate WebP filename in data.json: {video}")
        video_names.add(video)

        pairs = parse_codepoint_pairs(notation)
        translations = entry.get("translation") or {}
        compact_entry = {
            "id": record.get("id", ""),
            "video": video,
            "pairs": pairs,
            "translation": {
                language: translations.get(language) or {}
                for language in ("ZH-Yue", "ZH-HK", "EN-GB")
            },
        }
        entry_index = len(entries)
        entries.append(compact_entry)

        codepoints = {
            codepoint
            for pair in pairs
            for codepoint in pair
            if codepoint != 0
        }
        for codepoint in codepoints:
            by_codepoint.setdefault(str(codepoint), []).append(entry_index)

    handshapes = [None] + sorted(int(codepoint) for codepoint in by_codepoint)
    precomputed = {}
    for right_codepoint in handshapes:
        for left_codepoint in handshapes:
            candidate_indexes = set()
            for codepoint in (right_codepoint, left_codepoint):
                if codepoint is not None:
                    candidate_indexes.update(
                        by_codepoint.get(str(codepoint), ())
                    )

            ranked_matches = []
            for entry_index in candidate_indexes:
                rank = match_rank(
                    entries[entry_index], right_codepoint, left_codepoint
                )
                if rank is not None:
                    ranked_matches.append(
                        (rank, entries[entry_index]["id"], entry_index)
                    )
            ranked_matches.sort(
                key=lambda match: (
                    match[0],
                    entries[match[2]]["video"],
                )
            )

            key = (
                f"{right_codepoint if right_codepoint is not None else ''},"
                f"{left_codepoint if left_codepoint is not None else ''}"
            )
            precomputed[key] = [
                len(ranked_matches),
                [
                    entries[match[2]]["video"]
                    for match in ranked_matches[:PRECOMPUTED_MATCH_LIMIT]
                ],
            ]

    result = {
        "schema_version": LOOKUP_SCHEMA_VERSION,
        "entries": entries,
        "precomputed": precomputed,
    }
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(result, output_file, ensure_ascii=False, separators=(",", ":"))
        output_file.write("\n")

    print(
        f"Wrote {len(entries)} entries and {len(by_codepoint)} handshape indexes "
        f"to {OUTPUT_PATH}; skipped {missing_videos} entries without WebP videos."
    )


if __name__ == "__main__":
    build_lookup()
