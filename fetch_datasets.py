"""Download public document datasets and file them as real_*.pdf under data/raw_pdfs/<class>/.

Run on YOUR machine (needs internet):
    pip install datasets requests pillow
    python src/fetch_datasets.py --sroie --invoice2data --cord --rvl-invoice --fatura --cap 150

Sources (check each licence before submitting; record them in the report):
  --sroie         GitHub zzzDavid/ICDAR-2019-SROIE   626 real scanned receipts     (ICDAR-2019, research use)
  --invoice2data  GitHub invoice-x/invoice2data      13 real invoice/bill/receipt PDFs, mostly Dutch/French/German (MIT)
  --cord          HF naver-clova-ix/cord-v2          ~1000 receipt photos            (CC-BY-4.0)
  --rvl-invoice   HF chainyo/rvl-cdip-invoice        scanned invoices from RVL-CDIP  (research only)
  --fatura        HF mathieu1256/FATURA2-invoices    synthetic-but-realistic invoices, 50 layouts (CC-BY-NC-4.0)
--cap limits images taken PER SOURCE, so no single source swamps the others.
Status: --sroie and --invoice2data were tested; the Hugging Face sources (--cord, --rvl-invoice, --fatura) were NOT
testable where this was written (no access), so column names may need a small tweak (see `first_image_column`).
There is no large public dataset of real *bills*: collect your own bill PDFs/images and use import_images.py.
"""
import argparse, glob, io, os, shutil, sys, zipfile
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITHUB_ZIP = "https://codeload.github.com/{repo}/zip/refs/heads/{branch}"

# manual labels for the invoice2data sample PDFs (checked by reading their text)
INVOICE2DATA_LABELS = {"free_fiber.pdf": "bill", "oyo.pdf": "receipt"}   # everything else there is an invoice


def download(url, dest):
    import requests
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20): fh.write(chunk)


def save_pdf(img, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    img.convert("RGB").save(os.path.join(out_dir, name), "PDF", resolution=150)


def fetch_github_zip(repo, branch, tmp):
    zpath = os.path.join(tmp, repo.replace("/", "_") + ".zip")
    if not os.path.exists(zpath):
        print(f"downloading {repo} ..."); download(GITHUB_ZIP.format(repo=repo, branch=branch), zpath)
    with zipfile.ZipFile(zpath) as z: z.extractall(tmp)
    return os.path.join(tmp, f"{repo.split('/')[1]}-{branch}")


def do_sroie(raw, tmp, cap):
    root = fetch_github_zip("zzzDavid/ICDAR-2019-SROIE", "master", tmp)
    files = sorted(glob.glob(os.path.join(root, "data", "img", "*.jpg")))[:cap]
    for i, f in enumerate(files): save_pdf(Image.open(f), os.path.join(raw, "receipt"), f"real_receipt_sroie_{i:04d}.pdf")
    print(f"sroie: {len(files)} receipts")


def do_invoice2data(raw, tmp, cap):
    root = fetch_github_zip("invoice-x/invoice2data", "master", tmp)
    n = 0
    for f in sorted(glob.glob(os.path.join(root, "tests", "**", "*.pdf"), recursive=True))[:cap]:
        label = INVOICE2DATA_LABELS.get(os.path.basename(f), "invoice")
        os.makedirs(os.path.join(raw, label), exist_ok=True)
        shutil.copy(f, os.path.join(raw, label, f"real_{label}_i2d_{n:03d}.pdf")); n += 1
    print(f"invoice2data: {n} PDFs")


def first_image_column(ds):
    from datasets import Image as HFImage
    for name, feat in ds.features.items():
        if isinstance(feat, HFImage): return name
    raise RuntimeError(f"no image column in {list(ds.features)}")


def do_hf(raw, cap, repo, label, tag, split_pref=("train", "test", "validation")):
    from datasets import load_dataset
    ds = load_dataset(repo)
    split = next(s for s in split_pref if s in ds); ds = ds[split]
    col = first_image_column(ds)
    n = min(cap, len(ds))
    for i in range(n): save_pdf(ds[i][col], os.path.join(raw, label), f"real_{label}_{tag}_{i:04d}.pdf")
    print(f"{tag}: {n} {label}s from {repo}")


def main():
    ap = argparse.ArgumentParser()
    for flag in ["sroie", "invoice2data", "cord", "rvl-invoice", "fatura"]: ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--cap", type=int, default=150, help="max documents per source")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw_pdfs"))
    ap.add_argument("--tmp", default=os.path.join(ROOT, "data", "_downloads"))
    a = ap.parse_args(); os.makedirs(a.tmp, exist_ok=True)
    if not any([a.sroie, a.invoice2data, a.cord, a.rvl_invoice, a.fatura]): sys.exit("choose at least one source flag (see --help)")
    if a.sroie: do_sroie(a.out, a.tmp, a.cap)
    if a.invoice2data: do_invoice2data(a.out, a.tmp, a.cap)
    if a.cord: do_hf(a.out, a.cap, "naver-clova-ix/cord-v2", "receipt", "cord")
    if a.rvl_invoice: do_hf(a.out, a.cap, "chainyo/rvl-cdip-invoice", "invoice", "rvl")
    if a.fatura: do_hf(a.out, a.cap, "mathieu1256/FATURA2-invoices", "invoice", "fatura")
    print("done. Next: cd src && python build_dataset.py && python train.py")


if __name__ == "__main__":
    main()
