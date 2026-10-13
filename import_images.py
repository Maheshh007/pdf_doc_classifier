"""Convert a folder of document images (jpg/png/tif...) into scanned PDFs and file them under a class.

Usage (from the project folder):
    python src/import_images.py "C:/path/to/High Quality Invoice Image OCR" invoice
    python src/import_images.py "C:/path/to/receipts" receipt --limit 300
Files are written to data/raw_pdfs/<class>/<prefix>_<class>_<n>.pdf. Prefix `real_` = genuine documents (used for the honest\nreal-data tests); `pub_` = public but generated/templated datasets (e.g. Kaggle 'High Quality Invoice Image OCR').
Then run:  cd src && python build_dataset.py && python train.py
"""
import argparse, os, sys
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("label", choices=["invoice", "bill", "receipt"])
    ap.add_argument("--limit", type=int, default=0, help="max images to import (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="pick the --limit images at random instead of the first N")
    ap.add_argument("--prefix", choices=["real", "pub"], default="real",
                    help="real = genuine real-world documents; pub = public dataset that is generated/templated")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "data", "raw_pdfs", args.label); os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(args.folder) for f in fs if os.path.splitext(f)[1].lower() in EXTS)
    if args.shuffle:
        import random; random.Random(0).shuffle(files)
    if args.limit: files = files[:args.limit]
    if not files: sys.exit(f"No images found under {args.folder}")
    start = len([f for f in os.listdir(out_dir) if f.startswith(f"{args.prefix}_{args.label}_")])
    done = skipped = 0
    for i, path in enumerate(files, start):
        try:
            Image.open(path).convert("RGB").save(os.path.join(out_dir, f"{args.prefix}_{args.label}_{i}.pdf"), "PDF", resolution=150)
            done += 1
        except Exception as exc:
            skipped += 1; print(f"skipped {path}: {exc}")
    print(f"{done} PDFs written to {out_dir} ({skipped} skipped)")


if __name__ == "__main__":
    main()
