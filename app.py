"""Document Intelligence - PDF Classifier (Streamlit UI).
Presentation only: all processing lives in src/classify.py (which wraps the OCR + model pipeline)."""
import html
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import classify  # noqa: E402

st.set_page_config(page_title="Document Classification", layout="centered", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------- helpers
@st.cache_resource
def get_model():
    return classify.load_model()


CSS_PATH = os.path.join(ROOT, "ui", "styles.css")


@st.cache_data
def load_css(modified_time: float) -> str:
    """Cached per file version, so editing styles.css takes effect on the next rerun."""
    with open(CSS_PATH, encoding="utf-8") as fh:
        return fh.read()


def format_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def esc(value) -> str:
    return html.escape(str(value))


def reset_document():
    st.session_state.doc = None
    st.session_state.result = None
    st.session_state.uploader_key += 1


# ---------------------------------------------------------------- state
st.session_state.setdefault("doc", None)       # {"name", "data"}
st.session_state.setdefault("result", None)    # ClassificationResult
st.session_state.setdefault("uploader_key", 0)

# Stylesheet and header share one element so the stylesheet does not add an empty gap above the header.
st.markdown(f'<style>{load_css(os.path.getmtime(CSS_PATH))}</style><div class="app-header"><span class="product">DOCUMENT INTELLIGENCE</span>'
            '<span class="sep"></span><span class="module">PDF CLASSIFIER</span></div>', unsafe_allow_html=True)
st.markdown('<div class="page-title" role="heading" aria-level="1">Document Classification</div>'
            '<p class="page-sub">Extract document content and identify its financial document type.</p>', unsafe_allow_html=True)

if not os.path.exists(classify.MODEL_PATH):
    st.error("Classification model not found. Run `python src/train.py` from the project folder, then reload this page.")
    st.stop()
model = get_model()

# ---------------------------------------------------------------- upload panel
st.markdown('<div class="section-label">Upload document</div>', unsafe_allow_html=True)
with st.container(border=True):
    doc = st.session_state.doc
    if doc is None:
        uploaded = st.file_uploader("Upload document", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}",
                                    label_visibility="collapsed")
        st.markdown('<div class="upload-hint">PDF files only &middot; Maximum size: 200 MB</div>', unsafe_allow_html=True)
        if uploaded is not None:
            st.session_state.doc = {"name": uploaded.name, "data": uploaded.getvalue()}
            st.session_state.result = None
            st.rerun()
    else:
        info_col, action_col = st.columns([5, 1.2], vertical_alignment="center")
        info_col.markdown(f'<div class="file-row"><span class="name">{esc(doc["name"])}</span>'
                          f'<span class="meta">{format_size(len(doc["data"]))}</span></div>', unsafe_allow_html=True)
        action_col.button("Replace", on_click=reset_document, use_container_width=True)

        if st.session_state.result is None:
            stage_slot = st.empty()

            def show_stage(label: str):
                stage_slot.markdown(f'<div class="stage"><b>{esc(label)}</b></div>', unsafe_allow_html=True)

            st.session_state.result = classify.classify_document(doc["name"], doc["data"], model, on_stage=show_stage)
            if st.session_state.result.status in (classify.OK, classify.INCONCLUSIVE):
                stage_slot.markdown('<div class="stage"><b>Classification complete</b></div>', unsafe_allow_html=True)
            else:
                stage_slot.empty()

# ---------------------------------------------------------------- results
result = st.session_state.result
if result is not None:
    st.markdown('<div class="section-label" style="margin-top:28px">Classification result</div>', unsafe_allow_html=True)
    with st.container(border=True):
        failed = result.status in (classify.INVALID, classify.UNREADABLE, classify.ERROR)
        if result.status == classify.OK:
            badge = f'<span class="badge ok">{esc(result.category.capitalize())}</span>'
        elif result.status == classify.INCONCLUSIVE:
            badge = '<span class="badge warn">Inconclusive</span>'
        else:
            badge = '<span class="badge err">Not processed</span>'
        st.markdown(f'<div class="result-head">{badge}</div>', unsafe_allow_html=True)

        if result.message:
            tone = "err" if failed else "warn"
            st.markdown(f'<div class="notice {tone}">{esc(result.message)}</div>', unsafe_allow_html=True)

        rows = [("Document", esc(result.filename))]
        if result.status == classify.INCONCLUSIVE and result.category:
            rows.append(("Closest category", f"{esc(result.category.capitalize())} (low confidence)"))
        if result.confidence is not None:
            rows.append(("Model confidence", f"{result.confidence:.0%}"))
        if result.extraction_method in ("native", "ocr"):
            rows.append(("Text extraction", "Direct PDF text extraction" if result.extraction_method == "native"
                         else "OCR (Tesseract)"))
        if result.word_count:
            rows.append(("Words extracted", f"{result.word_count:,}"))
        if result.duration_s is not None:
            rows.append(("Processing time", f"{result.duration_s:.1f} s"))
        table = "".join(f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k, v in rows)
        st.markdown(f'<table class="fields">{table}</table>', unsafe_allow_html=True)

        if result.probabilities:
            bars = "".join(
                f'<div class="prob"><span class="lab">{esc(k.capitalize())}</span>'
                f'<span class="track"><span class="fill" style="display:block;width:{v * 100:.1f}%"></span></span>'
                f'<span class="val">{v:.0%}</span></div>' for k, v in result.probabilities.items())
            st.markdown('<div class="section-label" style="margin-top:18px">Class probabilities</div>'
                        f'<div class="probs">{bars}</div>', unsafe_allow_html=True)

    if result.text.strip():
        with st.expander("Extracted text"):
            st.text_area("Extracted text", result.text, height=300, disabled=True, label_visibility="collapsed")
