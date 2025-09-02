"""
Ascendion Survey Answering Assistant — single-file Streamlit app

What this does
- Single-session chat that drafts answers to EcoVadis, CDP, and Sedex SAQ questions
- Answers are grounded ONLY in uploaded Ascendion policy documents (RAG)
- Clear citations to file names/sections
- Export answers

How to run
1) Save this file as app.py
2) Python 3.10+
3) pip install -U streamlit openai tiktoken pypdf python-docx pandas numpy
   (Optional but helpful: pip install python-dotenv)
4) Set your OpenAI API key in env:  
   - macOS/Linux: export OPENAI_API_KEY="sk-..."  
   - Windows (PowerShell): $env:OPENAI_API_KEY="sk-..."
5) streamlit run app.py

Security note
- This prototype calls OpenAI APIs from the server-side (your machine). Do NOT expose your key in a browser.
- For production, add auth, logging/PII redaction, and move to a secured backend.

"""

import os
import io
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    st.stop()

try:
    import tiktoken
except Exception:
    tiktoken = None

from pypdf import PdfReader
from docx import Document as DocxDocument

# -----------------------------
# Config
# -----------------------------
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
MAX_CONTEXT_CHARS = 16_000  # conservative safety cap to keep prompts manageable

# -----------------------------
# Small utilities
# -----------------------------

def token_len(text: str, model: str = "gpt-4o-mini") -> int:
    if not tiktoken:
        # Fallback: rough estimate 4 chars ~ 1 token
        return max(1, len(text) // 4)
    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                chunks.append(buf)
            # If a single paragraph is huge, slice it
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(len(p), start + max_chars)
                    chunks.append(p[start:end])
                    start = end - overlap
                    if start < 0:
                        start = 0
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


def read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(texts)


def read_docx(file: io.BytesIO) -> str:
    doc = DocxDocument(file)
    return "\n".join([p.text for p in doc.paragraphs])


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set. Please set it in your environment.")
        st.stop()
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    # Batches help with rate limits
    out = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
    return np.vstack(out) if out else np.zeros((0, 1536), dtype=np.float32)


def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A: (n, d), b: (d,)
    if A.size == 0:
        return np.zeros((0,), dtype=np.float32)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return (A_norm @ b_norm).astype(np.float32)


# -----------------------------
# Survey styles & prompt system
# -----------------------------
SURVEY_TEMPLATES = {
    "EcoVadis": (
        "You are drafting an evidence-backed answer to an EcoVadis CSR/ESG questionnaire. "
        "Write concisely and factually, citing Ascendion documents by file name and section headers. "
        "Prefer bullets. Include policy excerpts only when directly relevant."
    ),
    "CDP": (
        "You are drafting a CDP (Carbon Disclosure Project) response. "
        "Use precise metrics, baselines, scopes, and governance language from sources. "
        "Map terms to CDP taxonomy when possible (e.g., Scope 1/2/3, risk/opportunity). "
        "Cite documents by file name and section."
    ),
    "Sedex SAQ": (
        "You are drafting a Sedex SAQ response. Use supplier ethics, labor standards, H&S, and compliance "
        "language from sources. Be practical, reference procedures and responsible roles. "
        "Cite documents by file name and section."
    ),
    "Generic": (
        "Draft a compliance-grade answer grounded only in the provided Ascendion sources. "
        "Be concise, avoid speculation, and include citations to file name and section."
    ),
}

SYSTEM_RULES = (
    "You are Ascendion's policy-grounded assistant. Answer ONLY using the provided context chunks. "
    "If the answer is not supported, say so and request the specific document to be uploaded. "
    "NEVER fabricate or guess. Include citations like [filename ➜ section/title]."
)

ANSWER_FORMAT_HINT = (
    "If numerical data are present, include units and timeframes. "
    "If policies describe processes, summarize roles, frequency, and escalation paths."
)


def build_prompt(survey_type: str, question: str, context_blocks: List[Tuple[str, str]]) -> str:
    style = SURVEY_TEMPLATES.get(survey_type, SURVEY_TEMPLATES["Generic"]) 
    context_text = []
    total_chars = 0
    for fname, chunk in context_blocks:
        snippet = f"\n\n— Source: {fname}\n{chunk.strip()}"
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            break
        context_text.append(snippet)
        total_chars += len(snippet)
    context = "".join(context_text)

    prompt = (
        f"{SYSTEM_RULES}\n\n"
        f"Survey type: {survey_type}. {style}\n\n"
        f"Question:\n{question}\n\n"
        f"Relevant sources:{context}\n\n"
        f"Guidance: {ANSWER_FORMAT_HINT}\n\n"
        f"Write the final answer now."
    )
    return prompt


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Ascendion Survey Answering Assistant", layout="wide")

st.title("Ascendion Survey Answering Assistant")
st.caption("Single-session RAG chat that answers EcoVadis / CDP / Sedex SAQ from uploaded policies only.")

with st.sidebar:
    st.header("1) Upload Ascendion policies")
    files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    survey_type = st.selectbox(
        "Survey format",
        ["EcoVadis", "CDP", "Sedex SAQ", "Generic"],
        index=0,
    )

    top_k = st.slider("# of context chunks", 3, 15, 6, step=1)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, step=0.05)
    model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL, help="e.g., gpt-4o-mini, gpt-4.1-mini, etc.")

    st.markdown("---")
    st.subheader("Export settings")
    include_metadata = st.checkbox("Include citations in export", value=True)

st.markdown("### 2) Ask a survey question")
question = st.text_area(
    "Paste a question from EcoVadis / CDP / Sedex SAQ",
    height=120,
    placeholder="e.g., Describe your organization's policies and due diligence processes related to supplier labor standards...",
)

colA, colB = st.columns([3, 2])
with colA:
    run_btn = st.button("Generate answer", type="primary", use_container_width=True)
with colB:
    clear_btn = st.button("Clear session", use_container_width=True)

if clear_btn:
    for k in ["kb_texts", "kb_meta", "kb_vectors", "last_answer", "history_df"]:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# -----------------------------
# Build knowledge base from uploads
# -----------------------------
if files:
    texts = []
    meta = []
    for f in files:
        ext = (f.name.split(".")[-1] or "").lower()
        try:
            if ext == "pdf":
                content = read_pdf(f)
            elif ext == "docx":
                content = read_docx(f)
            else:
                content = f.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")
            continue

        # Chunk and attach basic section titles (heuristic: first line of chunk)
        chunks = chunk_text(content)
        for ch in chunks:
            first_line = ch.strip().splitlines()[0][:100] if ch.strip().splitlines() else ""
            texts.append(ch)
            meta.append({"file": f.name, "section": first_line})

    if texts:
        client = get_openai_client()
        with st.spinner("Embedding uploaded documents..."):
            vectors = embed_texts(client, texts)
        st.session_state["kb_texts"] = texts
        st.session_state["kb_meta"] = meta
        st.session_state["kb_vectors"] = vectors
        st.success(f"Indexed {len(texts)} chunks from {len(files)} files.")
    else:
        st.info("No readable content found in uploads yet.")

# -----------------------------
# Answer generation
# -----------------------------
answer = None
if run_btn:
    if not question.strip():
        st.warning("Please paste a survey question.")
    elif "kb_vectors" not in st.session_state or st.session_state["kb_vectors"].size == 0:
        st.warning("Please upload Ascendion policies first.")
    else:
        client = get_openai_client()
        kb_vectors: np.ndarray = st.session_state["kb_vectors"]
        kb_texts: List[str] = st.session_state["kb_texts"]
        kb_meta: List[Dict[str, Any]] = st.session_state["kb_meta"]

        # Retrieve
        q_emb = client.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding
        q_emb = np.array(q_emb, dtype=np.float32)
        sims = cosine_sim_matrix(kb_vectors, q_emb)
        idx = np.argsort(-sims)[: top_k]

        context_blocks: List[Tuple[str, str]] = []
        for i in idx:
            fname = kb_meta[i]["file"]
            sec = kb_meta[i]["section"]
            chunk = kb_texts[i]
            header = f"{fname} ➜ {sec}" if sec else fname
            context_blocks.append((header, chunk))

        prompt = build_prompt(survey_type, question, context_blocks)

        with st.spinner("Drafting answer with GPT..."):
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_RULES},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = completion.choices[0].message.content
            st.session_state["last_answer"] = answer

# -----------------------------
# UI Output
# -----------------------------
if st.session_state.get("last_answer"):
    st.markdown("### Draft answer")
    st.write(st.session_state["last_answer"])  # renders markdown/bullets

    # Simple Q/A history for export
    hist_rows = st.session_state.get("history_df")
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "survey_type": survey_type,
        "question": question,
        "answer": st.session_state["last_answer"],
    }
    if hist_rows is None:
        hist_rows = pd.DataFrame([row])
    else:
        hist_rows = pd.concat([hist_rows, pd.DataFrame([row])], ignore_index=True)
    st.session_state["history_df"] = hist_rows

    st.markdown("---")
    st.subheader("Export")

    # Download answer as Markdown
    md_bytes = st.session_state["last_answer"].encode("utf-8")
    st.download_button(
        "Download this answer (.md)",
        data=md_bytes,
        file_name="ascendion_survey_answer.md",
        mime="text/markdown",
    )

    # Download Q/A history as CSV
    csv_bytes = st.session_state["history_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Q&A history (.csv)",
        data=csv_bytes,
        file_name="ascendion_survey_history.csv",
        mime="text/csv",
    )

# -----------------------------
# Helper panel: what was retrieved
# -----------------------------
if "kb_meta" in st.session_state and st.session_state["kb_meta"]:
    with st.expander("View indexed chunks (debug)"):
        meta_df = pd.DataFrame(st.session_state["kb_meta"])[:500]
        st.dataframe(meta_df, use_container_width=True)

st.markdown(
    """
    ---
    **Tips for high-quality outputs**
    - Upload the latest Ascendion policies: Code of Conduct, Supplier Code, ESG/CSR policy, Environmental policy, Health & Safety, Diversity & Inclusion, Anti-bribery/Corruption, Human Rights, Data Privacy/InfoSec (ISO), etc.
    - Add evidence docs (training logs, committee charters, KPIs) for stronger answers.
    - Ask ONE question at a time (single chat). Re-run for new questions.
    - The model will refuse to answer if evidence isn't found — upload the right doc and retry.
    """
)
