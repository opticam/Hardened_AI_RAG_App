"""Security guardrails: input sanitization, file validation, output checking."""

import os
import re
import logging
from datetime import datetime, timedelta

import streamlit as st

from config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE_MB,
    MAX_QUERIES_PER_SESSION,
    RATE_LIMIT_WINDOW_HOURS,
)

logger = logging.getLogger(__name__)

# ============================================================
# INPUT SANITIZATION
# ============================================================

_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions|prompts|rules)",
    r"(?i)you\s+are\s+now\s+(a|an|the)",
    r"(?i)new\s+(instructions|role|persona|identity)\s*:",
    r"(?i)(system|admin)\s+prompt\s*:",
    r"(?i)forget\s+(everything|all|your\s+rules|your\s+instructions)",
    r"(?i)disregard\s+(all|any|the|your)\s+(previous|above|prior)",
    r"(?i)override\s+(all|previous|your)\s+(instructions|rules|settings)",
    r"(?i)do\s+not\s+follow\s+(any|the|your)\s+(previous|above|prior)",
    r"(?i)pretend\s+(you\s+are|to\s+be)",
    r"(?i)act\s+as\s+(a|an|if)",
    r"(?i)jailbreak",
    r"(?i)DAN\s+mode",
]

_COMPILED_PATTERNS = [re.compile(p) for p in _INJECTION_PATTERNS]


def sanitize_text(text: str) -> str:
    """Strip common prompt injection patterns from text."""
    sanitized = text
    for pattern in _COMPILED_PATTERNS:
        sanitized = pattern.sub("[FILTERED]", sanitized)
    if sanitized != text:
        logger.warning("Prompt injection pattern detected and filtered.")
    return sanitized


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_uploaded_file(uploaded_file) -> bool:
    """Validate file type, size, and content before processing."""
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"⚠️ File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
        logger.warning(f"File rejected: size {uploaded_file.size} exceeds limit")
        return False

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        st.error(f"⚠️ Unsupported file type: `{{ext}}`. Allowed: {{ALLOWED_EXTENSIONS}}")
        logger.warning(f"File rejected: extension {ext} not allowed")
        return False

    try:
        import magic
        file_bytes = uploaded_file.getvalue()
        detected_mime = magic.from_buffer(file_bytes, mime=True)
        if detected_mime not in ALLOWED_MIME_TYPES:
            st.error(
                f"⚠️ File content type `{{detected_mime}}` doesn't match "
                f"expected types. The file may be corrupted or disguised."
            )
            logger.warning(f"File rejected: MIME {{detected_mime}} not in {{ALLOWED_MIME_TYPES}}")
            return False
    except ImportError:
        logger.info("python-magic not available, skipping MIME validation")

    if uploaded_file.size == 0:
        st.error("⚠️ The uploaded file is empty.")
        return False

    return True


# ============================================================
# OUTPUT VALIDATION
# ============================================================

_OUTPUT_RED_FLAGS = [
    "sk-",
    "api_key",
    "api key",
    "secret_key",
    "password",
    "access_token",
    "OPENAI_API_KEY",
    "system prompt is",
    "my instructions are",
]


def validate_response(response: str) -> str:
    """Check LLM output for signs of prompt injection success or data leakage."""
    response_lower = response.lower()
    for flag in _OUTPUT_RED_FLAGS:
        if flag.lower() in response_lower:
            logger.warning(f"Output validation triggered: response contained '{{flag}}'")
            return (
                "⚠️ I'm unable to provide that information. "
                "Please rephrase your question about the document."
            )
    return response


# ============================================================
# RATE LIMITING
# ============================================================

def init_rate_limiter():
    """Initialize rate limiting state in Streamlit session."""
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "first_query_time" not in st.session_state:
        st.session_state.first_query_time = None


def check_rate_limit() -> bool:
    """Check if the user has exceeded the rate limit."""
    init_rate_limiter()
    now = datetime.now()

    if (
        st.session_state.first_query_time
        and now - st.session_state.first_query_time
        > timedelta(hours=RATE_LIMIT_WINDOW_HOURS)
    ):
        st.session_state.query_count = 0
        st.session_state.first_query_time = None

    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        remaining = (
            st.session_state.first_query_time
            + timedelta(hours=RATE_LIMIT_WINDOW_HOURS)
            - now
        )
        minutes_left = int(remaining.total_seconds() / 60)
        st.error(
            f"⚠️ Rate limit reached ({{MAX_QUERIES_PER_SESSION}} queries). "
            f"Please try again in ~{{minutes_left}} minutes."
        )
        logger.warning("Rate limit exceeded")
        return False

    if st.session_state.first_query_time is None:
        st.session_state.first_query_time = now
    st.session_state.query_count += 1

    return True
