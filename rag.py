"""RAG pipeline: load, chunk, embed, store, retrieve, generate."""

import os
import logging

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    RETRIEVER_SEARCH_TYPE,
    RETRIEVER_K,
    SYSTEM_PROMPT,
)
from guardrails import sanitize_text

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================
# STEP 1: LOAD DOCUMENTS
# ============================================================
def load_documents(file_path: str):
    """Load a PDF or text file into LangChain Document objects."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {{file_path}}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {{file_path}}")

    documents = loader.load()

    if not documents:
        raise ValueError("No content could be extracted from the document.")

    logger.info(f"Loaded {{len(documents)}} page(s) from {{os.path.basename(file_path)}}")
    return documents


# ============================================================
# STEP 2: CHUNK DOCUMENTS
# ============================================================
def chunk_documents(documents):
    """Split documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "Document could not be split into chunks. "
            "The document may be empty or contain only whitespace."
        )

    logger.info(f"Created {{len(chunks)}} chunks (size={{CHUNK_SIZE}}, overlap={{CHUNK_OVERLAP}})")
    return chunks


# ============================================================
# STEP 3: CREATE VECTOR STORE
# ============================================================
def create_vector_store(chunks):
    """Embed document chunks and store in a FAISS vector database."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)

    logger.info(f"Vector store created with {{len(chunks)}} vectors using {{EMBEDDING_MODEL}}")
    return vector_store


# ============================================================
# STEP 4: BUILD THE RAG CHAIN
# ============================================================
def create_rag_chain(vector_store):
    """Create the Retrieval-Augmented Generation chain."""
    retriever = vector_store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": RETRIEVER_K},
    )

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    def format_docs(docs):
        logger.info(f"Retrieved {{len(docs)}} chunks for query")
        combined = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return sanitize_text(combined)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough() | sanitize_text,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain created successfully")
    return rag_chain


# ============================================================
# FULL PIPELINE
# ============================================================
def build_rag_pipeline(file_path: str):
    """Execute the full RAG pipeline: load -> chunk -> embed -> chain."""
    try:
        documents = load_documents(file_path)
        chunks = chunk_documents(documents)
        vector_store = create_vector_store(chunks)
        rag_chain = create_rag_chain(vector_store)

        logger.info("RAG pipeline built successfully")
        return rag_chain

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Pipeline validation error: {{e}}")
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {{type(e).__name__}}: {{e}}")
        raise RuntimeError(
            "Failed to process the document. Please check that the file "
            "is valid and try again."
        ) from e
