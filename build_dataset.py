"""Step 2/3: OCR every PDF, clean the text, build + audit the structured dataset (CSV)."""
import os, sys, hashlib
import pandas as pd
from joblib import Parallel, delayed
from ocr_utils import extract_text, clean_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw_pdfs")
MIN_WORDS = 8

def source_of(path):
    name = os.path.basename(path)
    return "real" if name.startswith("real_") else "public" if name.startswith("pub_") else "synthetic"


def file_md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def process(path, label):
    text, method, err = extract_text(path)
    return dict(file=os.path.relpath(path, RAW), label=label,
                source=source_of(path),
                extraction_method=method, error=err, raw_text=text, clean_text=clean_text(text), file_md5=file_md5(path))

def main():
    jobs = [(os.path.join(RAW, c, f), c) for c in ["invoice", "bill", "receipt"]
            for f in sorted(os.listdir(os.path.join(RAW, c))) if f.endswith(".pdf")]
    # Incremental build: OCR is slow, so reuse stored text for files already in dataset.csv
    # (same path; and same content when a hash was stored). Use --force to re-OCR everything.
    cache = {}
    prev_path = os.path.join(ROOT, "data", "dataset.csv")
    if os.path.exists(prev_path) and "--force" not in sys.argv:
        prev = pd.read_csv(prev_path)
        cache = {r["file"]: r for r in prev.to_dict("records")}
    rows, todo = [], []
    for path, label in jobs:
        rel = os.path.relpath(path, RAW)
        old = cache.get(rel)
        # only OCR results are worth caching; native extraction is instant (and always re-run)
        if old is not None and old["extraction_method"] == "ocr" and (pd.isna(old.get("file_md5", float("nan"))) or old["file_md5"] == file_md5(path)):
            text = "" if pd.isna(old["raw_text"]) else old["raw_text"]
            rows.append(dict(file=rel, label=label, source=old["source"], extraction_method=old["extraction_method"],
                             error="", raw_text=text, clean_text=clean_text(text), file_md5=file_md5(path)))
        else:
            todo.append((path, label))
    print(f"reusing {len(rows)} cached extractions, processing {len(todo)} new/changed files")
    rows += Parallel(n_jobs=2)(delayed(process)(p, l) for p, l in todo)
    df = pd.DataFrame(rows)
    df["n_words"] = df.clean_text.str.split().str.len()
    df["text_hash"] = df.clean_text.map(lambda t: hashlib.md5(t.encode()).hexdigest())

    # ---- quality audit (documented in the report) ----
    audit = {
        "total_files": len(df),
        "extraction_failed": int((df.extraction_method == "failed").sum()),
        "empty_or_too_short(<%d words)" % MIN_WORDS: int((df.n_words < MIN_WORDS).sum()),
        "duplicate_texts": int(df.duplicated("text_hash", keep="first").sum()),
    }
    # cross-label duplicate text = incorrectly labelled candidates
    conflict = df.groupby("text_hash").label.nunique()
    audit["label_conflicts(same text, different label)"] = int((conflict > 1).sum())
    print(audit)
    df.loc[(df.extraction_method == "failed") | (df.n_words < MIN_WORDS)].to_csv(
        os.path.join(ROOT, "reports", "rejected_documents.csv"), index=False,
        columns=["file", "label", "extraction_method", "n_words", "error"])
    clean = df[(df.extraction_method != "failed") & (df.n_words >= MIN_WORDS)]
    clean = clean[~clean.text_hash.isin(conflict[conflict > 1].index)].drop_duplicates("text_hash")
    clean.drop(columns=["text_hash"]).to_csv(os.path.join(ROOT, "data", "dataset.csv"), index=False)
    pd.Series(audit).to_json(os.path.join(ROOT, "reports", "dataset_audit.json"), indent=2)
    print(f"kept {len(clean)} / {len(df)} documents")
    print(clean.groupby(["label", "source", "extraction_method"]).size())

if __name__ == "__main__":
    main()
