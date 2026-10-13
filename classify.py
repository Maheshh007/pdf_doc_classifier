"""Backend for the app: validation + extraction + classification. No Streamlit imports here.
Reuses ocr_utils (extraction/cleaning/financial signal) and the saved sklearn pipeline unchanged."""
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import joblib

from ocr_utils import extract_text, clean_text, financial_signal

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "doc_classifier.joblib")
MAX_FILE_BYTES = 200 * 1024 * 1024      # must match server.maxUploadSize in .streamlit/config.toml
CONF_THRESHOLD = 0.55                   # below this top-class probability the result is "inconclusive"
MIN_WORDS = 8
MIN_FIN_TERMS = 2                       # fewer distinct finance terms -> probably not an invoice/bill/receipt

# result states
OK, INCONCLUSIVE, UNREADABLE, INVALID, ERROR = "ok", "inconclusive", "unreadable", "invalid", "error"


@dataclass
class ClassificationResult:
    status: str
    filename: str
    size_bytes: int
    message: str = ""                         # user-facing explanation for non-OK states
    category: Optional[str] = None            # best guess (also set for low-confidence results)
    confidence: Optional[float] = None
    probabilities: dict = field(default_factory=dict)
    extraction_method: Optional[str] = None   # "native" | "ocr"
    text: str = ""
    word_count: int = 0
    duration_s: Optional[float] = None


def load_model():
    return joblib.load(MODEL_PATH)


def validate_pdf(filename: str, data: bytes) -> Optional[str]:
    """Return an error message, or None if the file looks like an acceptable PDF."""
    if not filename.lower().endswith(".pdf"):
        return "Only PDF files are supported. Convert the document to PDF and upload it again."
    if len(data) == 0:
        return "The file is empty. Select a different document."
    if len(data) > MAX_FILE_BYTES:
        return "The file exceeds the 200 MB limit. Reduce its size or split the document."
    if not data[:1024].lstrip().startswith(b"%PDF"):
        return "The file does not appear to be a valid PDF. It may be renamed or corrupted."
    return None


def classify_document(filename: str, data: bytes, model, on_stage: Callable[[str], None] = lambda s: None) -> ClassificationResult:
    """Run validation -> extraction -> classification. `on_stage` is called just before each real step."""
    start = time.perf_counter()
    result = ClassificationResult(status=ERROR, filename=filename, size_bytes=len(data))

    on_stage("Validating document")
    problem = validate_pdf(filename, data)
    if problem:
        result.status, result.message = INVALID, problem
        return result

    on_stage("Extracting text")
    try:
        text, method, err = extract_text(data)
    except Exception as exc:  # extract_text is defensive, but never let the UI crash
        result.message = f"Text extraction failed unexpectedly ({type(exc).__name__}). Try a different copy of the file."
        return result
    result.text, result.extraction_method = text, method
    cleaned = clean_text(text)
    result.word_count = len(cleaned.split())

    if method == "failed":
        result.status = UNREADABLE
        result.message = ("The document could not be read; it may be corrupted or password protected"
                          + (f" ({err})" if err else "") + ". Try re-exporting the PDF or a different file.")
        result.duration_s = time.perf_counter() - start
        return result
    if result.word_count < MIN_WORDS:
        result.status = UNREADABLE
        result.message = ("Too little readable text was found" + (" after OCR" if method == "ocr" else "")
                          + ". If this is a scan, upload a higher-resolution or better-lit copy.")
        result.duration_s = time.perf_counter() - start
        return result

    on_stage("Classifying document")
    try:
        proba = model.predict_proba([cleaned])[0]
    except Exception as exc:
        result.message = f"Classification failed ({type(exc).__name__}). Retrain the model with src/train.py and try again."
        result.duration_s = time.perf_counter() - start
        return result
    if financial_signal(cleaned) < MIN_FIN_TERMS:
        # Not a financial document: class probabilities would be meaningless, so none are reported.
        result.status = INCONCLUSIVE
        result.message = ("The text contains almost no invoice, bill or receipt content (amounts, totals, tax, payment terms). "
                          "This may be a different type of document.")
    else:
        order = proba.argsort()[::-1]
        result.probabilities = {str(model.classes_[i]): float(proba[i]) for i in order}
        result.category, result.confidence = str(model.classes_[order[0]]), float(proba[order[0]])
        if result.confidence < CONF_THRESHOLD:
            result.status = INCONCLUSIVE
            result.message = "The model is not confident enough to assign a category. The document may not be an invoice, bill or receipt."
        else:
            result.status = OK
    result.duration_s = time.perf_counter() - start
    return result
