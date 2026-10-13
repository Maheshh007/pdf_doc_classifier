"""Text extraction: native PDF text first, Tesseract OCR fallback for scanned pages."""
import io
import re
import numpy as np
import cv2
import pdfplumber
import pypdfium2 as pdfium
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image

MIN_NATIVE_CHARS = 40   # fewer chars than this => treat PDF as scanned


def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Grayscale -> upscale small images -> denoise -> adaptive threshold."""
    img = np.array(pil_img.convert("L"))
    h, w = img.shape
    if max(h, w) < 1600:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    elif max(h, w) > 2400:   # huge phone photos: cap size so OCR stays fast
        s = 2400 / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = cv2.fastNlMeansDenoising(img, None, h=12)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(img)


def ocr_image(pil_img: Image.Image) -> str:
    # psm 6 = assume a uniform block of text (good for receipts/invoices)
    return pytesseract.image_to_string(preprocess_for_ocr(pil_img), config="--oem 3 --psm 6")


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def extract_text(pdf_source, max_pages: int = 3, dpi: int = 170):
    """Extract text, then strip control characters (some PDFs emit NUL for unmapped glyphs, which breaks CSV storage)."""
    text, method, err = _extract_text_raw(pdf_source, max_pages, dpi)
    return _CONTROL_CHARS.sub("", text), method, err


def _extract_text_raw(pdf_source, max_pages: int = 3, dpi: int = 170):
    """Return (text, method, error). pdf_source = path or bytes.
    method in {"native", "ocr", "failed"}; never raises."""
    data = pdf_source if isinstance(pdf_source, (bytes, bytearray)) else open(pdf_source, "rb").read()
    native = ""
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            native = "\n".join((p.extract_text() or "") for p in pdf.pages[:max_pages])
        if len(native.strip()) >= MIN_NATIVE_CHARS:
            return native, "native", ""
    except Exception:
        native = ""
    try:
        pdf = pdfium.PdfDocument(data)
        parts = []
        for i in range(min(len(pdf), max_pages)):
            img = pdf[i].render(scale=dpi / 72).to_pil()
            parts.append(ocr_image(img))
        return "\n".join(parts), "ocr", ""
    except Exception as e:
        return native, "failed", f"{type(e).__name__}: {e}"


def clean_text(text: str) -> str:
    """Lower-case, drop non-printable chars, normalise numbers/whitespace."""
    t = text.lower()
    t = re.sub(r"[^\x20-\x7e\n]", " ", t)
    t = re.sub(r"\d{4,}", " longnum ", t)
    t = re.sub(r"\$?\d+[.,]\d{2}\b", " money ", t)
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


FIN_TERMS = {"total", "subtotal", "tax", "vat", "gst", "amount", "due", "invoice", "bill", "receipt", "payment", "paid",
             "cash", "change", "balance", "qty", "quantity", "price", "account", "visa", "mastercard", "card", "usage",
             "charges", "statement", "customer", "tend", "debit", "credit", "money"}


def financial_signal(cleaned: str) -> int:
    """How many distinct finance-related terms appear. Documents that are not invoices/bills/receipts
    score ~0-1, so the app can answer 'inconclusive' instead of forcing one of the three classes."""
    return len(FIN_TERMS & set(cleaned.split()))
