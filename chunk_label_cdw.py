#!C:\Users\VHACONPeterJ\labeler\.labeler\Scripts\python.exe
"""
chunk_label_cdw.py

Features:
- Reads cleaned CSV (must include Findings column).
- Labels text using labeler_fast.py.
- PROCESSING SAFETY: Saves a partial CSV every --save-every records (e.g., 1500).
- PROGRESS UPDATES: Prints speed/status every --progress-every records (e.g., 10).
- AUTO-MERGE: Combines all partial files into one master file at the end.

Example:
  python chunk_label_cdw.py --infile cxr_cleaned_chunk0001.csv --outfile-base cxr_labeled_chunk0001 --save-every 1500
"""

import argparse
import time
import os
from pathlib import Path
import pandas as pd

# -------------------------------------------------------------------------
# IMPORT YOUR LABELER
# -------------------------------------------------------------------------
# Ensure labeler_fast.py is in the same directory
import labeler_fast as L

LABEL_KEYS = [
    "any_opacity", "consolidation", "left_cons", "right_cons",
    "bilateral_consolidation", "pneumonia", "edema", "atelectasis",
    "r_atelectasis", "l_atelectasis", "bi_atelectasis", "mass",
    "pneumothorax", "effusion", "heart_failure", "cardiomegaly",
    "devices", "intubation", "fracture",
]

def _label_one_report_doc(report_doc) -> dict:
    """Labels a single spaCy Doc object."""
    per_sentence_labels = []

    for sent_span in report_doc.sents:
        if not sent_span.text.strip():
            continue

        # Get clause texts
        # Note: If split_into_clauses_doc needs a doc, use sent_span.as_doc()
        clause_texts = [
            c.strip()
            for c in L.split_into_clauses_doc(sent_span.as_doc())
            if c and c.strip()
        ]

        if not clause_texts:
            continue

        clause_labels = []
        # Parse clauses in batch for speed
        for clause_text, clause_doc in zip(
            clause_texts, L.nlp.pipe(clause_texts, batch_size=64)
        ):
            label = L.detect_bilateral_consolidation_doc(clause_doc)

            # Lateral logic
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
    # Ensure all keys exist
    return {k: int(agg.get(k, 0)) for k in LABEL_KEYS}

def main():
    parser = argparse.ArgumentParser(description="Batch label CDW reports with chunk saving and merging.")
    parser.add_argument("--infile", required=True, help="Path to input CSV")
    parser.add_argument("--outfile-base", required=True, help="Base name for output files (e.g., 'my_results')")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--findings-col", default="Findings", help="Column name for text")
    
    # Batching and Saving settings
    parser.add_argument("--save-every", type=int, default=1500, help="Save a partial CSV every N records")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N records")
    parser.add_argument("--batch-size", type=int, default=32, help="spaCy NLP pipe batch size")
    parser.add_argument("--no-merge", action="store_true", help="Skip merging files at the end")
    
    args = parser.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    # 1. Initialize Labeler
    print(f"Loading model '{args.model}'...")
    L.init(args.model)

    # 2. Load Data
    print(f"Reading {infile}...")
    df = pd.read_csv(infile)
    
    if args.findings_col not in df.columns:
        raise ValueError(f"Column '{args.findings_col}' not found in CSV.")

    # Prepare texts
    texts = df[args.findings_col].fillna("").astype(str).tolist()
    n_total = len(df)
    print(f"Total rows to process: {n_total}")

    # 3. Processing Loop
    saved_filenames = []  # To keep track of parts for merging
    
    # Global timing stats
    t0_all = time.perf_counter()
    t_last_report = t0_all
    
    # We iterate through the data in large "save chunks"
    for chunk_idx, start_row in enumerate(range(0, n_total, args.save_every)):
        end_row = min(start_row + args.save_every, n_total)
        
        # Slices for this chunk
        chunk_texts = texts[start_row:end_row]
        chunk_df_rows = df.iloc[start_row:end_row].copy()
        
        chunk_results = []
        
        print(f"\n--- Starting Chunk {chunk_idx+1}: Rows {start_row} to {end_row} ---")

        # Process this chunk using nlp.pipe
        # We enumerate so we can print progress updates within the chunk
        doc_stream = L.nlp.pipe(chunk_texts, batch_size=args.batch_size)
        
        for i, doc in enumerate(doc_stream):
            # i is local to this chunk (0 to save_every)
            # global_index is overall position
            global_index = start_row + i + 1
            
            # Label
            labels = _label_one_report_doc(doc)
            chunk_results.append(labels)

            # PROGRESS REPORTING
            if global_index % args.progress_every == 0 or global_index == n_total:
                now = time.perf_counter()
                total_elapsed = now - t0_all
                avg_total = total_elapsed / global_index
                
                # Instantaneous speed (since last report)
                recent_elapsed = now - t_last_report
                recent_count = args.progress_every 
                # Correction for end of loop where count might differ
                if global_index == n_total: 
                    recent_count = global_index % args.progress_every or args.progress_every

                avg_recent = recent_elapsed / recent_count if recent_count else 0
                t_last_report = now
                
                print(
                    f"Processed {global_index}/{n_total} | "
                    f"Avg (Total): {avg_total:.3f}s | "
                    f"Avg (Last {recent_count}): {avg_recent:.3f}s"
                )

        # 4. Save this Chunk
        label_df = pd.DataFrame(chunk_results)
        # Reorder columns safely
        label_df = label_df.reindex(columns=LABEL_KEYS)
        
        # Combine original data with new labels
        out_df = pd.concat([chunk_df_rows.reset_index(drop=True), label_df], axis=1)
        
        # Generate filename: base_0001.csv, base_0002.csv, etc.
        part_filename = f"{args.outfile_base}_{chunk_idx+1:04d}.csv"
        out_df.to_csv(part_filename, index=False)
        saved_filenames.append(part_filename)
        
        print(f"Safe-saved part {chunk_idx+1}: {part_filename}")

    # 5. Final Merge
    total_time = time.perf_counter() - t0_all
    print(f"\nProcessing complete in {total_time:.1f}s.")
    
    if not args.no_merge and saved_filenames:
        print("Merging all parts into master file...")
        master_filename = f"{args.outfile_base}_MASTER.csv"
        
        # Read back all small CSVs and concatenate
        # (This is safer than holding one giant DF in memory the whole time)
        combined_df = pd.concat([pd.read_csv(f) for f in saved_filenames], ignore_index=True)
        combined_df.to_csv(master_filename, index=False)
        
        print(f"SUCCESS: Merged file written to {master_filename}")
        print("You can delete the partial _000X.csv files if the master looks good.")
    else:
        print("Skipping merge step (or no files saved).")

if __name__ == "__main__":
    main()
