"""Export the curated PIPPA subset so others can skip step 2.

`python -m storythread.publish SOURCE REVIEW OUTDIR` writes two files:

  OUTDIR/pippa-subset.jsonl  the approved conversations, in PIPPA's row shape
  OUTDIR/review.jsonl        the approved review rows (windows, cards, groups)

`data.py prepare` accepts them in place of the full PIPPA file and
`work/review.jsonl`, and rebuilds the same datasets. A conversation keeps
only the fields `prepare` reads, so the export is smaller than the source
rows. Rerunning on a non-empty OUTDIR is refused.
"""
import argparse
from pathlib import Path

from .data import rows, scene_of, write_rows

KEPT = ("bot_id", "bot_name", "bot_greeting", "bot_description", "conversation")


def trimmed(row):
    """Returns a PIPPA row cut to the fields that scene_of reads."""
    return {key: row.get(key) for key in KEPT}


def export(source, review, outdir):
    """Writes the approved subset and its review rows into outdir.

    source is a PIPPA JSONL file, review is a review file whose approved
    rows name the scenes to keep. Raises ValueError when an approved scene
    is not in source. Returns the number of scenes written.
    """
    approved = {r["scene_id"]: r for r in rows(review)
                if r.get("approved") is True}
    kept = {}
    for row in rows(source):
        found = scene_of(row)
        if found and found[0] in approved and found[0] not in kept:
            kept[found[0]] = trimmed(row)
    missing = sorted(set(approved) - set(kept))
    if missing:
        raise ValueError(f"Approved scenes not in source: {missing[:5]}")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "pippa-subset.jsonl", kept.values())
    write_rows(out / "review.jsonl", approved.values())
    return len(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="PIPPA JSONL, e.g. pippa_deduped.jsonl")
    parser.add_argument("review", help="review.jsonl with approved rows")
    parser.add_argument("outdir", help="directory for the two output files")
    args = parser.parse_args()
    print(f"{export(args.source, args.review, args.outdir)} scenes written")


if __name__ == "__main__":
    main()
