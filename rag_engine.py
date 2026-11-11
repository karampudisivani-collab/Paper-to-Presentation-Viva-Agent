"""
PDF ingestion and local-vector RAG engine.

PDF text extraction uses PyMuPDF. Scanned pages use Tesseract OCR.
ChromaDB stores local hashed word/ngram vectors, so document indexing does
not consume Gemini embedding quota or depend on an external embedding model.
"""

import hashlib
import os
from typing import List

import fitz
import pytesseract
from PIL import Image

from sklearn.feature_extraction.text import HashingVectorizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


class LocalHashEmbeddings:
    """
    Lightweight local vectorizer implementing LangChain's Embeddings interface.
    It creates fixed-size word and bigram vectors without a network call.
    This is lexical vector retrieval, not a transformer semantic embedding.
    """

    def __init__(self):
        self.vectorizer = HashingVectorizer(
            n_features=2048,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b",
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()

    def embed_query(self, text: str) -> List[float]:
        matrix = self.vectorizer.transform([text])
        return matrix.toarray()[0].astype(float).tolist()


def _extract_pages(pdf_path: str) -> List[Document]:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError("The uploaded PDF could not be found.")

    documents = []
    try:
        pdf = fitz.open(pdf_path)
        if pdf.is_encrypted:
            raise ValueError("This PDF is password-protected. Upload an unlocked PDF.")

        for page_index in range(len(pdf)):
            page = pdf.load_page(page_index)
            text = page.get_text("text").strip()

            # OCR only pages that have no meaningful selectable text.
            if len(text) < 35:
                try:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(1.8, 1.8),
                        alpha=False
                    )
                    image = Image.frombytes(
                        "RGB", [pix.width, pix.height], pix.samples
                    )
                    text = pytesseract.image_to_string(
                        image, config="--psm 3"
                    ).strip()
                except pytesseract.TesseractNotFoundError:
                    raise RuntimeError(
                        "Tesseract OCR is required for scanned pages. "
                        "On Mac run: brew install tesseract"
                    )

            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"page": page_index + 1}
                ))

        pdf.close()
    except Exception:
        raise

    if not documents:
        raise ValueError(
            "No readable text was extracted. The PDF may be blank, damaged, "
            "or scanned without Tesseract OCR installed."
        )
    return documents


def _split_pages(pages: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("The PDF did not produce any searchable text chunks.")
    return chunks


def ingest_pdf(pdf_path: str, pdf_bytes: bytes = None):
    """
    Ingest a PDF and return (Chroma vector store, extracted page count, chunk count).
    Chroma uses an ephemeral in-memory client to avoid local SQLite read-only and
    stale-lock problems. Streamlit keeps the vector store in session state.
    """
    pages = _extract_pages(pdf_path)
    chunks = _split_pages(pages)

    # Unique collection name prevents collisions between uploads/sessions.
    raw = pdf_bytes if pdf_bytes is not None else open(pdf_path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    collection_name = "paper_" + digest

    embeddings = LocalHashEmbeddings()
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        # No persist_directory: use Chroma's in-memory ephemeral client.
    )

    # Add modest batches to keep memory use predictable.
    batch_size = 32
    for start in range(0, len(chunks), batch_size):
        store.add_documents(chunks[start:start + batch_size])

    return store, len(pages), len(chunks)


def retrieve_context(store, query: str, k: int = 6) -> str:
    if store is None:
        return "No paper has been indexed. Upload and process a PDF first."
    if not query or not query.strip():
        return "Please enter a search query."

    docs = store.similarity_search(query.strip(), k=max(1, min(k, 20)))
    if not docs:
        return "No matching passages were found in the uploaded paper."

    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        parts.append("[PDF page {}]\n{}".format(page, doc.page_content.strip()))
    return "\n\n---\n\n".join(parts)
