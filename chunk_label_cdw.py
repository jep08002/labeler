"""
python chunk_label_cdw.py \
  --infile cleaned_reports.csv \
  --outfile-base cxr_labeled_chunk0001 \
  --save-every 1500
  ""

#!C:\Users\VHACONPeterJ\labeler\.labeler\Scripts\python.exe
import argparse
import time
from pathlib import Path
import pandas as pd
import labeler_fast as L

LABEL_KEYS = [
    "any_opacity", "consolidation", "left_cons", "right_cons",
    "bilateral_consolidation", "pneumonia", "edema", "atelectasis",
    "r_atelectasis", "l_atelectasis", "bi_atelectasis", "mass",
    "pneumothorax", "effusion", "heart_failure", "cardiomegaly",
    "devices", "intubation", "fracture",
]

def _label_one_report_doc(report_doc) -> dict:
    per_sentence_labels = []
    for sent_span in report_doc.sents:
        if not sent_span.text.strip():
            continue
        
        # Note: If split_into_clauses_doc is text-based, use sent_span.text
        # If it requires a doc, sent_span.as_doc() is correct
        clause_texts = [
            c.strip() for c in L.split_into_clauses_doc(sent_span.as_doc())
            if c and c.strip()
        ]
        
        if not clause_texts:
            continue

        clause_labels = []
        for clause_text, clause_doc in zip(clause_texts, L.nlp.pipe(clause_texts, batch_size=64)):
            label = L.detect_bilateral_consolidation_doc(clause_doc)

            for side in ["left", "right"]:
                if L.side_mentions_consolidation(clause_text, side):
                    cons_val = label.get("consolidation")
                    if cons_val in [1, -1]:
                        label[f"{side}_cons"] = cons_val

            clause_labels.append(label)
        per_sentence_labels.append(L.aggregate_labels(clause_labels))

    agg = L.aggregate_labels(per_sentence_labels)
    return {k: int(agg.get(k, 0)) for k in LABEL_KEYS}

def label_reports(findings_texts, batch_size: int) -> list[dict]:
    results = []
    for report_doc in L.nlp.pipe(findings_texts, batch_size=batch_size):
        results.append(_label_one_report_doc(report_doc))
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch label CDW reports with periodic saving.")
    parser.add_argument("--infile", required=True, help="Input CSV")
    parser.add_argument("--outfile-base", required=True, help="Base name (e.g. cxr_labeled_chunk0001)")
    parser.add_argument("--save-every", type=int, default=1500, help="Save a file every N reports")
    parser.add_argument("--batch-size", type=int, default=32, help="NLP pipe batch size")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--findings-col", default="Findings")
    args = parser.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    L.init(args.model)
    df = pd.read_csv(infile)
    
    # Clean/Prepare texts
    texts = df[args.findings_col].fillna("").astype(str).tolist()
    n_total = len(df)
    
    t0_all = time.perf_counter()
    processed_count = 0
    file_counter = 0

    # Process in chunks defined by 'save_every'
    for start_idx in range(0, n_total, args.save_every):
        file_counter += 1
        end_idx = min(start_idx + args.save_every, n_total)
        
        chunk_df = df.iloc[start_idx:end_idx].copy()
        chunk_texts = texts[start_idx:end_idx]
        
        print(f"--- Processing File Part {file_counter} (Rows {start_idx} to {end_idx}) ---")
        
        # Use nlp.pipe batching within this chunk
        labeled_results = label_reports(chunk_texts, batch_size=args.batch_size)
        
        # Format results
        label_df = pd.DataFrame(labeled_results).reindex(columns=LABEL_KEYS)
        out_df = pd.concat([chunk_df.reset_index(drop=True), label_df], axis=1)
        
        # Construct filename: base + _0001, _0002, etc.
        out_filename = f"{args.outfile_base}_{file_counter:04d}.csv"
        out_df.to_csv(out_filename, index=False)
        
        processed_count += len(chunk_df)
        elapsed = time.perf_counter() - t0_all
        print(f"Saved: {out_filename} | Total Processed: {processed_count}/{n_total}")
        print(f"Current Avg: {elapsed / processed_count:.3f}s/report\n")

    print(f"Done. All parts saved starting with {args.outfile_base}")

if __name__ == "__main__":
    main()
