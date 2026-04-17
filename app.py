"""Streamlit frontend for the RAG Document Q&A application."""

import gc
import os
import logging
import tempfile

import streamlit as st

from rag import build_rag_pipeline
from guardrails import (
    validate_uploaded_file,
    validate_response,
    check_rate_limit,
)
from config import ALLOWED_EXTENSIONS

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="📄 RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A")
st.caption("Upload a document and ask questions about it — powered by RAG.")

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# API KEY VALIDATION
# ============================================================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "⚠️ `OPENAI_API_KEY` is not set. "
        "Please configure it in your `.env` file or environment variables."
    )
    st.stop()

# ============================================================
# SIDEBAR — File Upload + Controls
# ============================================================
with st.sidebar:
    st.header("📁 Upload Document")

    allowed_types = [ext.lstrip(".") for ext in ALLOWED_EXTENSIONS]
    uploaded_file = st.file_uploader(
        "Choose a PDF or text file",
        type=allowed_types,
        help=f"Supported formats: {{', '.join(ALLOWED_EXTENSIONS)}}. Max 10MB.",
    )

    if uploaded_file and st.session_state.rag_chain is None:
        if validate_uploaded_file(uploaded_file):
            with st.spinner("🔄 Processing document..."):
                tmp_path = None
                try:
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    safe_suffix = ext if ext in ALLOWED_EXTENSIONS else ".txt"

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=safe_suffix,
                        dir=tempfile.gettempdir(),
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    st.session_state.rag_chain = build_rag_pipeline(tmp_path)
                    st.success("✅ Document processed! Ask questions below.")
                    logger.info(f"Document processed: {{uploaded_file.name}}")

                except Exception as e:
                    st.error(f"❌ {{e}}")
                    logger.error(f"Document processing failed: {{e}}")

                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    if st.session_state.rag_chain:
        st.success("🟢 Document loaded — ready for questions")
    else:
        st.info("🔵 Upload a document to get started")

    st.divider()

    if st.button("🔄 Reset Application"):
        st.session_state.rag_chain = None
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.first_query_time = None
        gc.collect()
        logger.info("Application state reset")
        st.rerun()

# ============================================================
# CHAT INTERFACE
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):

    if st.session_state.rag_chain is None:
        st.warning("⚠️ Please upload a document first using the sidebar.")

    elif not check_rate_limit():
        pass

    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    raw_response = st.session_state.rag_chain.invoke(prompt)

                safe_response = validate_response(raw_response)
                st.markdown(safe_response)

                logger.info(
                    f"Query processed successfully "
                    f"(query_count={{st.session_state.query_count}})"
                )

            except Exception as e:
                safe_response = (
                    "⚠️ Sorry, something went wrong while processing your question. "
                    "Please try again or rephrase your question."
                )
                st.markdown(safe_response)
                logger.error(f"Chain invocation error: {{type(e).__name__}}: {{e}}")

        st.session_state.messages.append(
            {"role": "assistant", "content": safe_response}
        )
