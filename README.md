# PDF Document Classification Using OCR and Machine Learning
Classifies a PDF as **Invoice**, **Bill** or **Receipt** and serves it through a Streamlit app.

## Run
```bash
sudo apt install tesseract-ocr            # (Windows: install Tesseract + add to PATH)
pip install -r requirements.txt
python src/prepare_pdfs.py                # 1. build PDF corpus
cd src && python build_dataset.py         # 2. OCR + clean -> data/dataset.csv   (~15 min on 1 CPU)
python train.py                           # 3. train + evaluate -> models/, reports/
cd .. && streamlit run app.py             # 4. UI
```

## 1. Dataset research & sources  (Deliverable: dataset summary + source links)
| Source | What it contains | How it is used | Licence / permission |
|---|---|---|---|
| Supplied archive 1 – CVAT receipt set (`trainingdata.solutions`), 19 receipt photos + polygon/box annotations | Real, noisy phone photos of store receipts | Converted to **scanned PDFs** → class `receipt` (real) | Check the licence on the page you downloaded it from and record it here |
| Supplied archive 2 – *Household energy bill data.csv* (1000 rows, Kaggle) | Tabular household features + `amount_paid`; **not documents** | Each row drives the numbers (usage, amount) of a **synthetic utility bill PDF** → class `bill` | Check Kaggle licence |
| Two real electricity bill PDFs supplied by the student (West Bengal WBSEDCL, Maharashtra MSEDCL) | Real utility bills with embedded text (the Maharashtra bill is mostly Marathi) | Class `bill` (real), files `real_bill_1/2.pdf` | Personal documents; confirm you may include them in the submission and consider redacting names/account numbers |
| Kaggle *High Quality Invoice Image OCR* (user-supplied sample of 150; **40 random images imported**) | Clean invoice images from ONE generated layout (fake names/addresses) | Class `invoice`, files `pub_invoice_*.pdf`, source=`public` | Check Kaggle licence. Treated as public-but-generated, NOT as real-world invoices. 40 only: copies of one template add no information |
| Synthetic PDFs (`src/prepare_pdfs.py`) | 110 invoices, 110 bills, 110 receipts, random vendors/layouts/dates; ~35 % rasterised with blur/noise/rotation to mimic scans | Fills the gap: the supplied data had **no invoices and no real bills** | Self-generated |

**Adding your own image datasets:** `python src/import_images.py "<folder>" invoice --limit 100 --shuffle` (add `--prefix pub` for generated/templated public datasets) converts images to PDFs (see the script header), then rerun `build_dataset.py` and `train.py`.

### Recommended real-data plan (`src/fetch_datasets.py`)
| Class | Best public source | Notes |
|---|---|---|
| Receipt | SROIE (626 scans, GitHub), CORD-v2 (~1000, CC-BY-4.0), WildReceipt | plenty available |
| Invoice | RVL-CDIP invoice class, FATURA2 (CC-BY-NC), Kaggle *High Quality Invoice Image OCR*, invoice2data samples (13 PDFs, MIT) | Kaggle/RVL/FATURA are hosted off GitHub |
| Bill | **no large public set** - use the student's *Household Monthly Electricity Bill* folder, own bills (redact personal data), municipal/telecom bill samples | the weakest class; budget time to collect these |

Keep the classes roughly balanced (use `--cap`). If one class is entirely real scans and another is entirely synthetic, the model can learn "scan noise = receipt" instead of document type; this is exactly why the two real bills were first predicted as receipts. Aim for real documents in all three classes, then drop the `syn_` files.

**Limitation – please read:** the only real documents are 18 receipt photos and 2 electricity bills. There are NO real invoices; all other invoices, bills and receipts are synthetic, so the hold-out scores are NOT a measure of real-world accuracy. Before submitting, add real documents (e.g. SROIE / CORD, RVL-CDIP, Kaggle invoice and utility-bill datasets) in `data/raw_pdfs/<class>/` named `real_*.pdf`, then re-run steps 2–3. `build_dataset.py` is incremental: only new files are OCR'd (`--force` re-OCRs everything).

## 2. OCR research & choice
| Approach | Pros | Cons |
|---|---|---|
| Embedded text (pdfplumber) | Exact, instant | Empty for scans |
| **Tesseract 5 + OpenCV preprocessing (chosen fallback)** | Free, offline, good on printed text, easy to deploy | Weak on crumpled/low-res photos |
| EasyOCR / PaddleOCR | Better on photos | Heavy (PyTorch/Paddle), slow on CPU |
| Cloud OCR (Textract, Vision) | Best accuracy | Cost, privacy, needs keys |

Pipeline (`src/ocr_utils.py`): try native text → if < 40 chars treat as scan → render page (pypdfium2) → grayscale, upscale/limit size, denoise, adaptive threshold → Tesseract (`--oem 3 --psm 6`). Failures are caught and returned as `method="failed"` instead of crashing.

## 3. Dataset creation & preprocessing
`src/build_dataset.py` → `data/dataset.csv` (file, label, source, extraction_method, raw_text, clean_text, n_words).
Cleaning: lower-case, strip non-ASCII, mask long numbers (`longnum`), prices (`money`) and other digits (`num`), remove punctuation.
Audit (`reports/dataset_audit.json`, `reports/rejected_documents.csv`): 393 files → 1 extraction failure (corrupt PDF), 1 empty (blank page), 2 duplicate texts, 1 cross-label text conflict (possible mislabel) → **390 kept**: 112 bills (2 real), 150 invoices (40 public/generated, 110 synthetic), 128 receipts (18 real). Extracted text has control characters stripped (a Marathi bill returned NUL characters that truncated the CSV).

## 4. Model
TF-IDF (1–2-grams, sublinear tf) + classifier. Compared Logistic Regression, calibrated Linear SVM, Multinomial Naive Bayes by 5-fold CV macro-F1; Logistic Regression selected (ties on score; gives well-behaved probabilities for the "inconclusive" threshold and interpretable weights). Saved as one sklearn Pipeline (`models/doc_classifier.joblib`) so preprocessing travels with the model.

## 5. App (`app.py`, `src/classify.py`, `ui/styles.css`, `.streamlit/config.toml`)
Dark enterprise UI. `app.py` is presentation only; `src/classify.py` holds validation -> extraction -> classification (no Streamlit imports) and wraps the unchanged `ocr_utils.py` and saved model.
Shown only when real: filename and size, predicted category, model probability (Logistic Regression `predict_proba`, not calibrated on real data), extraction method (direct text / OCR), word count, measured processing time, class probabilities, full extracted text (expander).
Validation: `.pdf` extension, non-empty, <= 200 MB (matches `server.maxUploadSize`), `%PDF` header. Inconclusive/not-processed messages for: corrupt or unreadable PDF, < 8 words, fewer than 2 finance-related terms, top probability < 55 %.

## 6. Results (`reports/metrics.json`)
Real documents get a sample weight of 5 during training; sources are `real`, `public` (generated public datasets) and `synthetic`.
* 5-fold CV macro-F1: 0.994 ± 0.011 (LogReg), 0.995 (LinearSVC), 0.997 (NaiveBayes); LogReg kept because the differences are noise and its probabilities suit the app's confidence threshold.
* 80/20 hold-out (78 docs): accuracy 1.000. Optimistic: 95 % of documents are synthetic or generated from a few templates.
* **Real-data tests (the ones that matter, 20 real docs: 18 receipts + 2 bills):**
  * Train on synthetic + public only → test real docs: **14/20 (70 %)**. Receipts 14/18, bills 0/2. Errors are mostly real documents predicted as *invoice* (`reports/real_transfer_synthetic_only.csv`).
  * Leave-one-out over the 20 real docs (each predicted by a model that never saw it, real docs weighted): **19/20 (95 %)**. Receipts 18/18, bills 1/2; the Maharashtra bill (mostly Marathi) is called a receipt (`reports/real_leave_one_out.csv`).
  * There are no genuinely real invoices in the data, so invoice accuracy on real-world invoices has NOT been measured. Only 2 real bills exist, so bill results are anecdotal.
* The saved model is trained on everything, including the real documents, so it classifies those correctly by construction.

**Honest interpretation:** adding 40 single-template invoices did not help real-world generalisation: with synthetic+public training only, real receipts and bills were more often predicted as invoice (70 % vs 85-95 % before). The model works well on documents that look like its training templates and is unreliable on unfamiliar real layouts. The fix is more real, varied documents in all three classes (especially bills and invoices), not more copies of the same template.

## 7. Error analysis & improvements
* Likely real-world errors: restaurant "bills" (receipt-like), pro-forma/credit notes (invoice-like), phone-bill vs invoice for businesses, low-quality photos with garbled OCR.
* Improve: real multi-source data; layout features (LayoutLM / Donut); character n-grams to survive OCR noise; an explicit "other" class; better OCR (PaddleOCR) for photos; confidence calibration on real validation data.
