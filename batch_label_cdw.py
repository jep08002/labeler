#!/usr/bin/env python3
"""
batch_label_cdw.py

Reads the cleaned CSV output from cdw_report_preprocess.py (must include a Findings column),
runs each Findings text through your labeler, and writes an output CSV that contains the
original columns plus one column per label.

Features:
- Batch processing via spaCy nlp.pipe() for speed (labeling logic unchanged).
- Progress update every N reports (default: 10).
- Average time per report printed every N reports.
- Optional max-records limit for quick testing.

Example:
  python batch_label_cdw.py --infile cleaned_reports.csv --outfile labeled_reports.csv

Optional:
  python batch_label_cdw.py --infile cleaned_reports.csv --outfile labeled_reports.csv --max-records 1000 --progress-every 10 --batch-size 64
"""

import argparse
import time
from pathlib import Path

import pandas as pd

# IMPORTANT:
# This assumes your labeler module file is named labeler_fast.py (or change import below).
# It must provide:
#   - init(model_name: str) -> model
#   - nlp (global set by init)
#   - split_into_clauses_doc(doc)
#   - detect_bilateral_consolidation_doc(doc)
#   - side_mentions_consolidation(text, side)
#   - aggregate_labels(list_of_label_dicts)
#
# If your file name differs, change this import line.
import labeler_fast as L


# Columns (labels) your labeler aggregates/returns
LABEL_KEYS = [
    "any_opacity",
    "consolidation",
    "left_cons",
    "right_cons",
    "bilateral_consolidation",
    "pneumonia",
    "edema",
    "atelectasis",
    "r_atelectasis",
    "l_atelectasis",
    "bi_atelectasis",
    "mass",
    "pneumothorax",
    "effusion",
    "heart_failure",
    "cardiomegaly",
    "devices",
    "intubation",
    "fracture",
]


def _label_one_report_doc(report_doc) -> dict:
    """
    Label a single report Doc using the same logic you’ve been using:
    - Iterate sentences
    - Split into clauses
    - Parse each clause once (dependencies needed)
    - Aggregate labels
    """
    per_sentence_labels = []

    for sent_span in report_doc.sents:
        if not sent_span.text.strip():
            continue

        clause_texts = [
            c.strip()
            for c in L.split_into_clauses_doc(sent_span.as_doc())
            if c and c.strip()
        ]
        if not clause_texts:
            continue

        clause_labels = []
        # Parse clauses in a batch for this sentence for speed
        for clause_text, clause_doc in zip(
            clause_texts, L.nlp.pipe(clause_texts, batch_size=64)
        ):
            label = L.detect_bilateral_consolidation_doc(clause_doc)

            # Laterality post-processing (text-based) - unchanged
            if L.side_mentions_consolidation(clause_text, "left"):
                if label.get("consolidation") == 1:
                    label["left_cons"] = 1
                elif label.get("consolidation") == -1:
                    label["left_cons"] = -1

            if L.side_mentions_consolidation(clause_text, "right"):
                if label.get("consolidation") == 1:
                    label["right_cons"] = 1
                elif label.get("consolidation") == -1:
                    label["right_cons"] = -1

            clause_labels.append(label)

        per_sentence_labels.append(L.aggregate_labels(clause_labels))

    agg = L.aggregate_labels(per_sentence_labels)

    # Ensure every label key exists
    out = {k: int(agg.get(k, 0)) for k in LABEL_KEYS}
    return out


def label_reports(findings_texts, batch_size: int) -> list[dict]:
    """
    Label a list of findings strings. Uses nlp.pipe over reports for sentence segmentation speed.
    Clause parsing is still required per sentence (dependency logic), but we batch those too.
    """
    results: list[dict] = []

    # Parse whole reports in batches (for sents)
    for report_doc in L.nlp.pipe(findings_texts, batch_size=batch_size):
        results.append(_label_one_report_doc(report_doc))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch label cleaned CDW CXR reports CSV (adds label columns)."
    )
    parser.add_argument("--infile", required=True, help="Path to cleaned input CSV")
    parser.add_argument("--outfile", required=True, help="Path to output labeled CSV")
    parser.add_argument(
        "--model", default="en_core_web_sm", help="spaCy model name to load"
    )
    parser.add_argument(
        "--findings-col",
        default="Findings",
        help="Column containing report text to label (default: Findings)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N reports (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="spaCy pipe batch size for report docs (default: 32)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional maximum number of rows to process (for testing)",
    )
    args = parser.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)

    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    # Initialize labeler once
    # If you want a free speed win and you’re not using NER, you can disable NER inside labeler init.
    L.init(args.model)

    # Load CSV
    df = pd.read_csv(infile)

    if args.findings_col not in df.columns:
        raise ValueError(
            f"Findings column '{args.findings_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if args.max_records is not None:
        df = df.iloc[: args.max_records].copy()

    n_total = len(df)
    if n_total == 0:
        raise ValueError("No rows found to process.")

    # Prepare texts (ensure strings; empty -> "")
    texts = (
        df[args.findings_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # Process with progress
    labeled_rows: list[dict] = []
    t0_all = time.perf_counter()
    t_last_report = time.perf_counter()

    # We do chunked processing ourselves so we can print every progress-every
    step = max(1, int(args.progress_every))
    batch_size = max(1, int(args.batch_size))

    processed = 0
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        chunk_texts = texts[start:end]

        # Label this chunk
        chunk_labels = label_reports(chunk_texts, batch_size=batch_size)
        labeled_rows.extend(chunk_labels)

        processed = end

        # Progress reporting every N (or at end)
        if processed % step == 0 or processed == n_total:
            now = time.perf_counter()
            elapsed = now - t0_all
            avg_per = elapsed / processed if processed else float("nan")

            # Also show "recent" avg since last report point (optional but handy)
            recent_elapsed = now - t_last_report
            recent_n = step if (processed % step == 0) else (processed % step)
            recent_avg = recent_elapsed / recent_n if recent_n else avg_per
            t_last_report = now

            print(
                f"Processed {processed}/{n_total} | "
                f"avg/report (overall): {avg_per:.3f}s | "
                f"avg/report (last {recent_n}): {recent_avg:.3f}s"
            )

    # Attach label columns
    label_df = pd.DataFrame(labeled_rows)
    # Ensure ordering
    label_df = label_df.reindex(columns=LABEL_KEYS)

    out_df = pd.concat([df.reset_index(drop=True), label_df], axis=1)

    # Write output
    out_df.to_csv(outfile, index=False)

    total_elapsed = time.perf_counter() - t0_all
    print(f"Done. Wrote: {outfile}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Average time per report: {total_elapsed / n_total:.3f}s")


if __name__ == "__main__":
    main()
