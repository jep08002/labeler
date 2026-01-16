"""
python cxr_preprocess.py \
  --infile /path/to/raw_dump.txt \
  --outdir /path/to/output_chunks \
  --chunk-size 10000
  --base-name va_cxr_cleaned
  """
  
import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, List, Iterator

# ---------- block splitting ----------
SEP_LINE_RE = re.compile(r"^\s*#{3,}\s*$")  # line like ##### (or ###)

# ---------- section headers ----------
FINDINGS_HDR_RE = re.compile(r"(?im)^\s*findi\w*\s*[:/]\s*$")
IMPRESSION_HDR_RE = re.compile(r"(?im)^\s*impressi\w*\s*[:/]\s*$")

# Inline headers (when Findings: is on same line as content)
FINDINGS_INLINE_RE = re.compile(r"(?i)findi\w*\s*:\s*")
IMPRESSION_INLINE_RE = re.compile(r"(?i)impressi\w*\s*:\s*")

# Delete any line containing "history:" (case-insensitive), even if it's the last line (no trailing newline)
HISTORY_LINE_RE = re.compile(r"(?im)^.*\bhistory\s*:.*(?:\r?\n|$)")

# ---------- ID + datetime patterns ----------
ICN_RE = re.compile(r"\b(\d{9,12})\b")

DT_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}:\d{2})\b")
DT_US_RE = re.compile(
    r"\b(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\b",
    re.IGNORECASE,
)

def squash_commas_and_quotes(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    s = re.sub(r'"{2,}', '"', s)
    s = re.sub(r"(?m)^\s*[,\"]+\s*$", "", s)
    s = re.sub(r"(?:\s*,\s*){2,}", " ", s)
    s = re.sub(r'(?m)^\s*"+\s*', "", s)
    s = re.sub(r'(?m)\s*"+\s*$', "", s)

    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def normalize_dt(dt: str) -> str:
    dt = (dt or "").strip()
    dt = re.sub(r'"{2,}', '"', dt)
    dt = dt.strip('"').strip().strip(",")
    return dt

def first_nonempty_line(lines: List[str]) -> Tuple[int, Optional[str]]:
    for i, ln in enumerate(lines):
        if ln.strip():
            return i, ln
    return -1, None

def parse_header_line(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    icn = None
    m = ICN_RE.search(line)
    if m:
        icn = m.group(1)

    dts: List[str] = []
    for mm in DT_ISO_RE.finditer(line):
        dts.append(normalize_dt(mm.group(1)))
    for mm in DT_US_RE.finditer(line):
        v = normalize_dt(mm.group(1))
        if v not in dts:
            dts.append(v)

    exam_dt = dts[0] if len(dts) >= 1 else None
    report_dt = dts[1] if len(dts) >= 2 else None
    return icn, exam_dt, report_dt

def extract_findings_impression(report_body: str) -> str:
    t = report_body.replace("\r\n", "\n").replace("\r", "\n")
    t = squash_commas_and_quotes(t)

    # Remove any line containing "history:"
    t = HISTORY_LINE_RE.sub("", t)

    f_match = next(FINDINGS_HDR_RE.finditer(t), None)
    i_match = next(IMPRESSION_HDR_RE.finditer(t), None)

    if f_match:
        start_f = f_match.end()
        if i_match and i_match.start() > start_f:
            findings = t[start_f:i_match.start()].strip()
            impression = t[i_match.end():].strip()
            out = (findings + "\n\nIMPRESSION:\n" + impression).strip()
            return squash_commas_and_quotes(out)
        return squash_commas_and_quotes(t[start_f:].strip())

    f_inline = FINDINGS_INLINE_RE.search(t)
    if f_inline:
        tail = t[f_inline.end():]
        i_inline = IMPRESSION_INLINE_RE.search(tail)
        if i_inline:
            findings = tail[:i_inline.start()].strip()
            impression = tail[i_inline.end():].strip()
            out = (findings + "\n\nIMPRESSION:\n" + impression).strip()
            return squash_commas_and_quotes(out)
        return squash_commas_and_quotes(tail.strip())

    return squash_commas_and_quotes(t)

def iter_blocks_streaming(infile: Path, encoding: str = "utf-8") -> Iterator[Tuple[int, str]]:
    """
    Stream blocks separated by delimiter lines like '#####'.
    Yields (block_index_1based, block_text).
    """
    block_lines: List[str] = []
    idx = 0

    with infile.open("r", encoding=encoding, errors="ignore") as f:
        for line in f:
            # Normalize newlines; keep content without trailing '\n' then re-join with '\n'
            line_stripped_nl = line.rstrip("\n")

            if SEP_LINE_RE.match(line_stripped_nl):
                if block_lines:
                    idx += 1
                    yield idx, "\n".join(block_lines).strip("\n")
                    block_lines = []
                else:
                    # consecutive separators; ignore
                    continue
            else:
                block_lines.append(line_stripped_nl)

        # flush final block
        if block_lines:
            idx += 1
            yield idx, "\n".join(block_lines).strip("\n")

def chunk_path(outdir: Path, base_stem: str, chunk_num: int) -> Path:
    return outdir / f"{base_stem}_chunk{chunk_num:04d}.csv"

def open_new_chunk(outdir: Path, base_stem: str, chunk_num: int):
    path = chunk_path(outdir, base_stem, chunk_num)
    f = path.open("w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["PatientICN", "ExamDateTime", "ReportDateTime", "Findings"])
    return path, f, w

def main():
    ap = argparse.ArgumentParser(description="Preprocess CXR dump (##### blocks) into chunked clean CSVs for labeling.")
    ap.add_argument("--infile", required=True, help="Input raw file")
    ap.add_argument("--outdir", required=True, help="Output folder for chunked CSVs")
    ap.add_argument("--base-name", default="", help="Base output name (stem). Default: infile stem.")
    ap.add_argument("--chunk-size", type=int, default=10000, help="Rows written per chunk file (default 10000)")
    ap.add_argument("--encoding", default="utf-8", help="Input file encoding (default utf-8)")
    ap.add_argument("--debug-block", type=int, default=0, help="Print debug for block N (1-based). 0 disables.")
    args = ap.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_stem = args.base_name.strip() or infile.stem

    blocks_found = 0
    wrote_total = 0
    skipped = 0

    chunk_num = 1
    rows_in_chunk = 0

    chunk_csv_path, chunk_f, chunk_w = open_new_chunk(outdir, base_stem, chunk_num)

    try:
        for idx, block in iter_blocks_streaming(infile, encoding=args.encoding):
            blocks_found += 1

            lines = block.replace("\r\n", "\n").replace("\r", "\n").splitlines()
            header_i, header_line = first_nonempty_line(lines)

            if header_line is None:
                skipped += 1
                continue

            icn, exam_dt, report_dt = parse_header_line(header_line)

            # report body is everything after first non-empty header line
            body = "\n".join(lines[header_i + 1:]) if header_i >= 0 else "\n".join(lines)
            body = squash_commas_and_quotes(body)

            findings = extract_findings_impression(body)

            if args.debug_block == idx:
                print("\n========== DEBUG BLOCK", idx, "==========")
                print("HEADER LINE:")
                print(header_line[:300])
                print("PARSED:", icn, exam_dt, report_dt)
                print("\nBODY (first 300 chars):")
                print(body[:300])
                print("\nFINDINGS OUT (first 300 chars):")
                print((findings or "")[:300])
                print("========== END DEBUG ==========\n")

            if not icn:
                skipped += 1
                continue

            if not findings or not re.search(r"[A-Za-z0-9]", findings):
                skipped += 1
                continue

            # rotate chunk if needed (based on rows WRITTEN)
            if rows_in_chunk >= args.chunk_size:
                chunk_f.close()
                chunk_num += 1
                rows_in_chunk = 0
                chunk_csv_path, chunk_f, chunk_w = open_new_chunk(outdir, base_stem, chunk_num)

            chunk_w.writerow([icn, exam_dt or "", report_dt or "", findings])
            rows_in_chunk += 1
            wrote_total += 1

            # optional lightweight progress
            if wrote_total % 10000 == 0:
                print(f"Progress: wrote {wrote_total:,} rows (current chunk {chunk_num:04d}, rows in chunk {rows_in_chunk:,})")

    finally:
        try:
            chunk_f.close()
        except Exception:
            pass

    print("Done.")
    print(f"Blocks found : {blocks_found:,}")
    print(f"Wrote rows   : {wrote_total:,}")
    print(f"Skipped      : {skipped:,}")
    print(f"OUTDIR       : {outdir.resolve()}")
    print(f"Last chunk   : {chunk_num:04d} ({rows_in_chunk:,} rows)")
    print(f"Example file : {chunk_path(outdir, base_stem, 1).resolve()}")

if __name__ == "__main__":
    main()