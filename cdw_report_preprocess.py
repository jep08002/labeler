import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, List

# ---------- block splitting ----------
SEP_RE = re.compile(r"(?m)^\s*#{3,}\s*$")  # lines like ##### (or ###)

# ---------- section headers ----------
FINDINGS_HDR_RE = re.compile(r"(?im)^\s*findi\w*\s*:?\s*$")
IMPRESSION_HDR_RE = re.compile(r"(?im)^\s*impressi\w*\s*:?\s*$")

# Inline headers (when Findings: is on same line as content)
FINDINGS_INLINE_RE = re.compile(r"(?i)findi\w*\s*:\s*")
IMPRESSION_INLINE_RE = re.compile(r"(?i)impressi\w*\s*:\s*")

# ---------- ID + datetime patterns ----------
ICN_RE = re.compile(r"\b(\d{9,12})\b")

DT_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}:\d{2})\b")
DT_US_RE = re.compile(
    r"\b(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\b",
    re.IGNORECASE,
)

def squash_commas_and_quotes(s: str) -> str:
    """
    Aggressive cleaning for your exported format:
    - removes lines that are only commas/quotes
    - collapses repeated commas/quotes
    """
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # collapse repeated quotes
    s = re.sub(r'"{2,}', '"', s)

    # remove lines that are just commas/quotes/spaces
    s = re.sub(r"(?m)^\s*[,\"]+\s*$", "", s)

    # remove obvious ",,," junk anywhere
    s = re.sub(r"(?:\s*,\s*){2,}", " ", s)

    # trim stray quotes at line edges
    s = re.sub(r'(?m)^\s*"+\s*', "", s)
    s = re.sub(r'(?m)\s*"+\s*$', "", s)

    # normalize whitespace
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
    """
    Extract ICN + first two datetime strings from a single line.
    """
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
    """
    Prefer Findings + Impression sections if present.
    Otherwise, if Findings exists, return everything after Findings.
    Otherwise return cleaned body.
    """
    t = report_body.replace("\r\n", "\n").replace("\r", "\n")
    t = squash_commas_and_quotes(t)

    # Try clean line-based headers first
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

    # Inline "Findings:" fallback
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

def parse_blocks(raw: str):
    blocks = [b for b in SEP_RE.split(raw) if b.strip()]
    for idx, b in enumerate(blocks, start=1):
        yield idx, b

def main():
    ap = argparse.ArgumentParser(description="Preprocess CXR dump (##### blocks) into clean CSV for labeling.")
    ap.add_argument("--infile", required=True, help="Input raw file")
    ap.add_argument("--outfile", required=True, help="Output cleaned CSV path")
    ap.add_argument("--debug-block", type=int, default=0, help="Print debug for block N (1-based). 0 disables.")
    args = ap.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)

    raw = infile.read_text(encoding="utf-8", errors="ignore")

    wrote = 0
    skipped = 0
    blocks_found = 0

    with outfile.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PatientICN", "ExamDateTime", "ReportDateTime", "Findings"])

        for idx, block in parse_blocks(raw):
            blocks_found += 1

            lines = block.replace("\r\n", "\n").replace("\r", "\n").splitlines()
            header_i, header_line = first_nonempty_line(lines)

            if header_line is None:
                skipped += 1
                continue

            icn, exam_dt, report_dt = parse_header_line(header_line)

            # KEY FIX: report body is EVERYTHING AFTER the first non-empty header line
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

            w.writerow([icn, exam_dt or "", report_dt or "", findings])
            wrote += 1

    print("Done.")
    print(f"Blocks found : {blocks_found}")
    print(f"Wrote rows   : {wrote}")
    print(f"Skipped      : {skipped}")
    print(f"OUTFILE      : {outfile.resolve()}")

if __name__ == "__main__":
    main()
