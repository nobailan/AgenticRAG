"""
data_prepare.py -- Data preparation pipeline for the Agentic RAG system.

Reads PDFs from the veracier-industries dataset, extracts text
(direct extraction for searchable PDFs, OCR for scanned/mixed PDFs),
chunks text, generates embeddings, and builds FAISS + BM25 indexes.

Usage:
    python data_prepare.py                    # Full pipeline
    python data_prepare.py --skip-ocr         # Skip OCR for scanned PDFs
    python data_prepare.py --limit 50         # Process only first 50 PDFs
    python data_prepare.py --output-dir ./my_indexes  # Custom output directory

Outputs (in project root by default):
    chunks.jsonl      — One JSON object per line, each with chunk_id, text, metadata
    faiss.index       — FAISS IndexFlatIP (cosine similarity via inner product)
    bm25_index.pkl    — Pickled rank_bm25.BM25Okapi instance
"""

# Load .env file FIRST, before any other project imports, so that
# RAG_EMBEDDING_MODEL and other env vars are set when config.py initializes.
import env_loader
env_loader.load_env()

import argparse
import json
import logging
import os
import pickle
import re
import sys
from itertools import count
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from config import config

logger = logging.getLogger(__name__)


# =============================================================================
# Type aliases
# =============================================================================

ChunkDict = Dict[str, Any]
"""A single chunk dict with keys: chunk_id (str), text (str), metadata (dict)."""


# =============================================================================
# Step 1: Load master index
# =============================================================================

def load_master_index() -> pd.DataFrame:
    """Load MASTER_INDEX.csv and return a DataFrame.

    The CSV maps each doc_id to its filename, entity, language, format,
    pages, classification, and other metadata.

    Returns:
        DataFrame with columns:
            doc_id, question_id, role, entity, filename, classification,
            language, format, pages, description
    """
    path = config.master_index_path
    if not path.exists():
        raise FileNotFoundError(f"MASTER_INDEX.csv not found at {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded MASTER_INDEX: {len(df)} rows, {df['doc_id'].nunique()} unique doc_ids")
    return df


def get_unique_pdfs(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate MASTER_INDEX to one row per unique PDF.

    Since a PDF can appear in multiple question contexts (same doc_id, same filename),
    we deduplicate on doc_id, keeping the first occurrence.

    Args:
        df: DataFrame from load_master_index().

    Returns:
        DataFrame with one row per unique doc_id.
    """
    unique = df.drop_duplicates(subset="doc_id", keep="first").reset_index(drop=True)
    logger.info(f"Deduplicated to {len(unique)} unique PDFs")
    return unique


# =============================================================================
# Step 2: PDF text extraction
# =============================================================================

def extract_text_searchable(pdf_path: Path) -> Optional[str]:
    """Extract text from a searchable PDF using PyMuPDF (fitz).

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Extracted text as a single string, or None if extraction fails.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Install with: pip install pymupdf")
        return None

    try:
        doc = fitz.open(str(pdf_path))
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages_text.append(f"[Page {page_num + 1}]\n{text}")
        doc.close()

        if pages_text:
            return "\n\n".join(pages_text)
        else:
            logger.warning(f"No searchable text in: {pdf_path}")
            return None
    except Exception as e:
        logger.error(f"PyMuPDF failed on {pdf_path}: {e}")
        return None


def extract_text_scanned(pdf_path: Path, language: str = "eng") -> Optional[str]:
    """Extract text from a scanned (image-only) PDF using Tesseract OCR.

    Uses pdf2image to render pages to images, then pytesseract for OCR.

    Args:
        pdf_path: Absolute path to the PDF file.
        language: Tesseract language code (e.g., 'fra', 'deu', 'fra+eng').

    Returns:
        Extracted text as a single string, or None if OCR fails.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        logger.error("pdf2image and/or pytesseract not installed. "
                       "Install with: pip install pdf2image pytesseract")
        return None

    # ---- Configure paths for poppler and tesseract on Windows (conda env) ----
    import platform, sys as _sys
    if platform.system() == "Windows":
        # Always derive prefix from Python executable location
        # Structure: <conda_root>/python.exe or <conda_root>/envs/<name>/python.exe
        _py_exe = Path(_sys.executable)
        _py_parents = _py_exe.parents  # 0=python dir, 1=env name dir, 2=envs dir, 3=conda root
        if len(_py_parents) >= 3 and _py_parents[1].name == "envs":
            # Running from a named env: E:/anaconda3/envs/graph/python.exe
            _env_prefix = str(_py_parents[0])  # E:/anaconda3/envs/graph
            _base_prefix = str(_py_parents[2])  # E:/anaconda3
        else:
            # Running from base or non-standard layout
            _env_prefix = str(_py_parents[0])
            _base_prefix = _env_prefix

        # Search: env-specific Library/bin first, then base
        _search_paths = [
            Path(_env_prefix) / "Library" / "bin",
            Path(_base_prefix) / "Library" / "bin",
        ]
        _tessdata_search = [
            Path(_env_prefix) / "share" / "tessdata",
            Path(_base_prefix) / "share" / "tessdata",
        ]

        _poppler_path = None
        for _sp in _search_paths:
            if _sp.exists() and (_sp / "pdftoppm.exe").exists():
                _poppler_path = str(_sp)
                _tesseract_exe = str(_sp / "tesseract.exe")
                if Path(_tesseract_exe).exists():
                    pytesseract.pytesseract.tesseract_cmd = _tesseract_exe
                    logger.debug(f"Tesseract configured: {_tesseract_exe}")
                break

        for _td in _tessdata_search:
            if _td.exists():
                os.environ.setdefault("TESSDATA_PREFIX", str(_td) + os.sep)
                logger.debug(f"TESSDATA_PREFIX set to: {_td}")
                break
    else:
        _poppler_path = None
    # -------------------------------------------------------------------------

    try:
        # poppler_path is needed on Windows when poppler is not in system PATH
        poppler_kwargs = {}
        if _poppler_path:
            poppler_kwargs["poppler_path"] = _poppler_path

        images = convert_from_path(
            str(pdf_path),
            dpi=config.ocr_dpi,
            first_page=1,
            **poppler_kwargs,
        )
    except Exception as e:
        logger.error(f"pdf2image failed on {pdf_path}: {e}")
        return None

    pages_text = []
    for i, image in enumerate(images, 1):
        try:
            # Map our language names to Tesseract 3-letter codes
            tess_lang = _map_language_to_tesseract(language)
            text = pytesseract.image_to_string(image, lang=tess_lang)
            if text.strip():
                pages_text.append(f"[Page {i}]\n{text}")
        except Exception as e:
            logger.warning(f"Tesseract OCR failed on page {i} of {pdf_path}: {e}")

    return "\n\n".join(pages_text) if pages_text else None


def _map_language_to_tesseract(lang: str) -> str:
    """Map our language codes to Tesseract 3-letter codes.

    Args:
        lang: Language string from MASTER_INDEX (e.g., 'fr', 'fr/en', 'de').

    Returns:
        Tesseract language string (e.g., 'fra+eng', 'deu').
    """
    mapping = {
        "fr": "fra",
        "en": "eng",
        "de": "deu",
        "it": "ita",
        "es": "spa",
    }
    parts = lang.lower().split("/")
    tess_parts = [mapping.get(p, "eng") for p in parts]
    return "+".join(tess_parts)


def extract_text_from_pdf(
    pdf_path: Path,
    format_type: str,
    language: str,
) -> Optional[str]:
    """Extract text from a single PDF using the appropriate method.

    Strategy:
        1. 'searchable' → PyMuPDF direct text extraction.
        2. 'scanned'    → pdf2image + pytesseract OCR.
        3. 'mixed'      → PyMuPDF first, OCR fallback for pages with no text.

    Args:
        pdf_path: Absolute path to the PDF file.
        format_type: One of 'searchable', 'scanned', 'mixed'.
        language: Language code from MASTER_INDEX (e.g., 'fr', 'en', 'fr/en').

    Returns:
        Extracted text string, or None if all extraction methods fail.
    """
    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        return None

    if format_type == "searchable":
        text = extract_text_searchable(pdf_path)
        if text:
            return text
        # Fallback to OCR
        logger.info(f"Falling back to OCR for searchable PDF: {pdf_path}")
        return extract_text_scanned(pdf_path, language)

    elif format_type == "scanned":
        return extract_text_scanned(pdf_path, language)

    elif format_type == "mixed":
        # Try searchable first, fallback to OCR
        text = extract_text_searchable(pdf_path)
        if text and len(text) > 100:
            return text
        logger.info(f"Mixed PDF with insufficient searchable text, OCR fallback: {pdf_path}")
        return extract_text_scanned(pdf_path, language)

    else:
        logger.warning(f"Unknown format '{format_type}' for {pdf_path}, trying searchable...")
        text = extract_text_searchable(pdf_path)
        if text:
            return text
        return extract_text_scanned(pdf_path, language)


# =============================================================================
# Step 3: Text chunking (sentence-first — never breaks a sentence)
# =============================================================================

# Module-level counter for chunk IDs
_chunk_counter = count(0)


def _reset_chunk_counter(start: int = 0) -> None:
    """Reset the global chunk ID counter."""
    global _chunk_counter
    _chunk_counter = count(start)


# Language code → NLTK punkt model name mapping
_PUNKT_LANG_MAP = {
    "fr": "french",
    "en": "english",
    "de": "german",
    "it": "italian",
    "es": "spanish",
}


def _sent_tokenize(text: str, language: str = "en") -> List[str]:
    """Split text into sentences, language-aware.

    Tries NLTK's pre-trained punkt tokenizer first (handles abbreviations
    like Mr./Dr./S.A./GmbH correctly). Falls back to a regex-based
    sentence splitter for robustness offline.

    Args:
        text: Raw text to split.
        language: ISO 639-1 language code (fr/en/de/it/es).

    Returns:
        List of sentence strings (whitespace stripped, empty filtered).
    """
    # ---- Strategy 1: NLTK punkt (handles abbreviations natively) ----
    try:
        import nltk

        nltk_lang = _PUNKT_LANG_MAP.get(language, "english")
        # Ensure punkt is downloaded (idempotent once cached)
        try:
            nltk.data.find(f"tokenizers/punkt_tab/{nltk_lang}")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                nltk.download("punkt", quiet=True)

        raw_sentences = nltk.sent_tokenize(text, language=nltk_lang)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if sentences:
            return sentences
    except Exception:
        logger.debug("NLTK sentence tokenizer unavailable, using regex fallback")

    # ---- Strategy 2: Regex-based fallback ----
    return _regex_sent_tokenize(text)


# Common abbreviation patterns for each language.
# Used by the regex fallback to avoid splitting after these tokens.
# IMPORTANT: longer abbreviations MUST appear before shorter prefixes
# (e.g. S\.A\.R\.L before S\.A, Ste before St) to prevent premature matching.
_ABBREV_PATTERNS_PER_LANG: Dict[str, str] = {
    "en": r"(?:Mrs|Mr|Prof|Dr|Inc|Ltd|Co|Corp|U\.S|U\.K|"
          r"etc|vs|i\.e|e\.g|a\.m|p\.m|vol|dept|est|approx|Jr|Sr|Ms|St)",
    "fr": r"(?:S\.A\.R\.L|Mme|Mlle|S\.A|E\.U|Dr|Me|Pr|Ste|St|env|etc|"
          r"av|apr|not|art|chap|éd|vol|p\.ex|M)",
    "de": r"(?:GmbH|Dr|Prof|Dipl|Ing|Hr|Fr|AG|KG|z\.B|d\.h|evtl|usw|bzw|ca|ggf|"
          r"inkl|exkl|Nr|Jh|Abk)",
    "it": r"(?:Sig\.ra|S\.p\.A|S\.r\.l|Dott|Ing|Avv|Arch|Rag|Geom|Sig|"
          r"ecc|ca|cfr|n\.b|es)",
    "es": r"(?:S\.A\.de|Srta|S\.A|S\.L|Sra|Dr|Dra|Prof|Lic|Ing|Arq|Sr|"
          r"etc|aprox|ej|p\.ej|dpto|avda|ctra)",
}


def _ends_with_abbreviation(text: str) -> bool:
    """Check whether *text* ends with a known abbreviation (including trailing dot).

    Uses word-boundary (\\b) to prevent short abbreviations from matching
    the tails of ordinary words (e.g. 'es' matching 'employees').

    Args:
        text: Stripped text of a raw sentence part.

    Returns:
        True if the last word-token is a known abbreviation like Mr./Dr./S.A.
    """
    import re

    # Build a combined pattern from all languages
    combined = "|".join(
        f"(?:{p})" for p in _ABBREV_PATTERNS_PER_LANG.values()
    )
    # \b ensures the abbreviation is a standalone word-token (not the tail
    # of a longer word like "employees" matching Italian "es")
    # \. matches the abbreviation's own trailing period
    pattern = re.compile(rf"\b(?:{combined})\.$", re.IGNORECASE)

    return bool(pattern.search(text.strip()))


def _regex_sent_tokenize(text: str) -> List[str]:
    """Regex-based sentence splitter for European languages (standard re).

    Strategy (two-pass, avoids variable-width lookbehind):
        1. Split on every potential sentence boundary:
           punctuation (.!?。！？) + whitespace/newline + capital/CJK char.
        2. Merge back false splits where the previous part ends with
           a known abbreviation (Mr., Dr., S.A., GmbH, etc.).

    Also splits on double-newline (paragraph) boundaries first.

    Args:
        text: Raw text to split.

    Returns:
        List of sentence strings.
    """
    import re

    # ---- Pass 0: split into paragraphs ----
    paragraphs = re.split(r"\n\s*\n", text)

    # ---- Pass 1: split each paragraph on potential sentence boundaries ----
    # Pattern: punctuation, then whitespace/newline, then capital/CJK letter
    boundary_re = re.compile(
        r"(?<=[.!?。！？])"          # After sentence-ending punctuation
        r"[\s\n]+"                    # Whitespace or newline
        r"(?=[A-ZÀ-ÖØ-ÝА-Я一-鿿])"  # Followed by capital or CJK
    )

    raw_parts: List[str] = []
    for para in paragraphs:
        if not para.strip():
            continue
        split_parts = boundary_re.split(para)
        split_parts = [p.strip() for p in split_parts if p.strip()]
        raw_parts.extend(split_parts)

    if not raw_parts:
        return [text.strip()] if text.strip() else []

    # ---- Pass 2: merge abbreviation false splits ----
    # After a raw split, a part may end with an abbreviation (e.g. "Mr.",
    # "S.A."). The next part (a name or company name) would have been
    # wrongly separated. We merge them back.
    #
    # IMPORTANT: after merging, the combined text may STILL end with
    # another abbreviation (e.g. "M. ... S.A." → need to merge with the
    # next part too). We use a while-loop to handle recursive merges.
    sentences: List[str] = []
    i = 0
    while i < len(raw_parts):
        current = raw_parts[i].strip()
        i += 1

        # Recursive merge: keep absorbing next parts while current ends
        # with a known abbreviation (Mr., Dr., S.A., GmbH, etc.)
        while i < len(raw_parts) and _ends_with_abbreviation(current):
            current = current + " " + raw_parts[i].strip()
            i += 1

        if current:
            sentences.append(current)

    return sentences


# Cache for tiktoken encoder (initialized lazily)
_tiktoken_enc = None


def _get_token_counter():
    """Return a token-counting function (tiktoken or character fallback).

    Returns:
        Callable str → int.
    """
    global _tiktoken_enc
    if _tiktoken_enc is not None:
        return lambda t: len(_tiktoken_enc.encode(t))

    try:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        return lambda t: len(_tiktoken_enc.encode(t))
    except ImportError:
        # Fallback: ~4 chars ≈ 1 token for European languages
        return lambda t: len(t) // 4


def chunk_text(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[ChunkDict]:
    """Split text into overlapping chunks — NEVER breaks a sentence.

    Strategy (sentence-first):
        1. Tokenize text into sentences using language-aware NLTK punkt
           (or regex fallback). This correctly handles abbreviations like
           Mr., Dr., S.A., GmbH across EN/FR/DE/IT/ES.
        2. Greedily group sentences into chunks, keeping each chunk ≤
           chunk_size tokens. A sentence is NEVER split mid-way — it's
           either included whole in the current chunk (if it fits) or
           starts a new chunk.
        3. Overlap is at the SENTENCE level: the new chunk starts with
           as many complete sentences from the END of the previous chunk
           as fit within chunk_overlap tokens.
        4. Edge case: a single sentence longer than chunk_size tokens
           falls back to RecursiveCharacterTextSplitter for that sentence
           only, with the strongest separator being ". " (period+space).

    Each chunk gets chunk_id format: "doc_0", "doc_1", ...

    Args:
        text: Full extracted text from one PDF.
        metadata: Source metadata dict with keys:
            source_file, page, entity, language, format, classification.
        chunk_size: Max chunk size in tokens (default from config).
        chunk_overlap: Overlap in tokens (default from config).

    Returns:
        List of ChunkDict with keys: chunk_id, text, metadata.
    """
    if chunk_size is None:
        chunk_size = config.chunk_size
    if chunk_overlap is None:
        chunk_overlap = config.chunk_overlap

    if not text or not text.strip():
        logger.debug("Empty text, skipping chunking")
        return []

    token_counter = _get_token_counter()

    # ---- Step 1: sentence tokenization ----
    language = metadata.get("language", "en") if metadata else "en"
    sentences = _sent_tokenize(text, language)

    if not sentences:
        logger.debug("No sentences extracted, skipping chunking")
        return []

    # ---- Step 2: group sentences into chunks ----
    chunks: List[ChunkDict] = []
    current_sentences: List[str] = []
    current_tokens = 0

    # Pre-compute token counts for all sentences
    sent_tokens_list: List[int] = [token_counter(s) for s in sentences]

    for i, sentence in enumerate(sentences):
        sent_tokens = sent_tokens_list[i]

        # ---- Edge case: single sentence exceeds chunk_size ----
        if sent_tokens > chunk_size:
            # Flush any accumulated sentences first
            if current_sentences:
                chunks.append(_make_chunk(current_sentences, metadata))
                current_sentences = []
                current_tokens = 0

            # Split this long sentence with RecursiveCharacterTextSplitter
            for sub_chunk in _split_long_sentence(sentence, chunk_size, chunk_overlap, token_counter):
                chunks.append(_make_chunk([sub_chunk], metadata))
            continue

        # ---- Would adding this sentence overflow? ----
        if current_tokens + sent_tokens > chunk_size and current_sentences:
            # Finalize current chunk
            chunks.append(_make_chunk(current_sentences, metadata))

            # ---- Sentence-level overlap: carry over last N sentences
            # that fit within chunk_overlap tokens ----
            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tok = token_counter(s)
                if overlap_tokens + s_tok <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tok
                else:
                    break

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        # Add this sentence to the current chunk
        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Flush remaining sentences
    if current_sentences:
        chunks.append(_make_chunk(current_sentences, metadata))

    return chunks


def _make_chunk(sentences: List[str], metadata: Dict[str, Any]) -> ChunkDict:
    """Build a single ChunkDict from a list of complete sentences.

    Args:
        sentences: List of sentence strings (all must be non-empty).
        metadata: Source metadata dict.

    Returns:
        ChunkDict with chunk_id, text, metadata.
    """
    chunk_id = f"doc_{next(_chunk_counter)}"
    return {
        "chunk_id": chunk_id,
        "text": " ".join(sentences).strip(),
        "metadata": dict(metadata),
    }


def _split_long_sentence(
    sentence: str,
    chunk_size: int,
    chunk_overlap: int,
    token_counter,
) -> List[str]:
    """Split a single over-long sentence using RecursiveCharacterTextSplitter.

    Only triggered when a single sentence exceeds chunk_size tokens
    (rare for business documents with chunk_size=512). The strongest
    separator is ". " (period+space) so we still favor clause boundaries.

    Args:
        sentence: The sentence text that's too long.
        chunk_size: Max tokens per sub-chunk.
        chunk_overlap: Token overlap between sub-chunks.
        token_counter: Token counting function.

    Returns:
        List of sub-chunk strings.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=token_counter,
            separators=[". ", "。", "; ", "! ", "? ", ", ", " ", ""],
        )
        return splitter.split_text(sentence)
    except ImportError:
        # Last resort: hard cut by character count
        char_limit = chunk_size * 4
        sub_chunks = []
        for i in range(0, len(sentence), char_limit - chunk_overlap * 4):
            sub_chunks.append(sentence[i:i + char_limit])
        return sub_chunks


# =============================================================================
# Step 4: Embedding generation
# =============================================================================

def generate_embeddings(
    chunks: List[ChunkDict],
    model_name: str | None = None,
) -> Tuple[np.ndarray, List[ChunkDict]]:
    """Generate dense embeddings for all chunks using SentenceTransformers.

    Args:
        chunks: List of chunk dicts (each must have "text" key).
        model_name: HuggingFace model name (default from config).

    Returns:
        Tuple of (embeddings matrix [N, dim] as float32, list of chunk dicts).
        The chunk dicts are enriched with an 'embedding_idx' field.
    """
    if model_name is None:
        model_name = config.embedding_model_name

    if not chunks:
        logger.warning("No chunks to embed")
        return np.empty((0, config.embedding_dim), dtype=np.float32), []

    from sentence_transformers import SentenceTransformer

    # Resolve device explicitly and log it
    device = config.embedding_device
    logger.info(f"Loading embedding model: {model_name} (device={device})")
    try:
        model = SentenceTransformer(model_name, device=device)
    except OSError as e:
        logger.error(
            f"Failed to load embedding model '{model_name}': {e}\n"
            f"  Possible causes:\n"
            f"  1. No internet connection to HuggingFace (https://huggingface.co)\n"
            f"  2. Model not cached locally\n"
            f"  Solutions:\n"
            f"  - Use --embedding-model with a locally available model\n"
            f"  - Download the model manually to the HuggingFace cache\n"
            f"  - Set HF_HUB_OFFLINE=1 if the model is already cached"
        )
        raise

    # Verify the device actually used by the model
    actual_device = str(model.device) if hasattr(model, 'device') else 'unknown'
    logger.info(f"Embedding model loaded on device: {actual_device}")

    # Update config's embedding_dim to match the actual model
    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim != config.embedding_dim:
        logger.warning(
            f"Embedding dimension mismatch: model={actual_dim}, "
            f"config={config.embedding_dim}. Using model dimension."
        )
        config.embedding_dim = actual_dim

    texts = [c["text"] for c in chunks]
    total_chunks = len(texts)
    batch_size = 32

    logger.info(
        "Embedding %d chunks on %s (model=%s, batch_size=%d) — this may take a while...",
        total_chunks, actual_device, model_name, batch_size,
    )

    # Manual batching with per-CHUNK progress bar (not per-batch).
    # This ensures the tqdm total matches the chunk count (e.g. 4750) rather
    # than the batch count (e.g. 149) or document count (e.g. 1100), so the
    # user can see real per-chunk progress.
    all_embeddings: List[np.ndarray] = []
    with tqdm(total=total_chunks, desc="Embedding chunks", unit="chunk") as pbar:
        for i in range(0, total_chunks, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = model.encode(
                batch_texts,
                normalize_embeddings=True,  # L2 norm → cosine via inner product
                show_progress_bar=False,
            )
            all_embeddings.append(batch_emb)
            pbar.update(len(batch_texts))

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # Tag each chunk with its embedding index
    for i, chunk in enumerate(chunks):
        chunk["embedding_idx"] = i

    logger.info(f"Generated embeddings: shape={embeddings.shape}, "
                 f"dtype={embeddings.dtype}")
    return embeddings, chunks


# =============================================================================
# Step 5: FAISS index
# =============================================================================

def build_faiss_index(
    embeddings: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Build and save a FAISS IndexFlatIP (cosine similarity) index.

    Embeddings are assumed to be L2-normalized, so inner product (IP)
    equals cosine similarity.

    Args:
        embeddings: [N, dim] numpy float32 array.
        output_path: Where to save the index (default from config).
    """
    if output_path is None:
        output_path = config.faiss_index_path

    if embeddings.size == 0:
        logger.warning("Empty embeddings, creating empty FAISS index")
        embeddings = np.zeros((0, config.embedding_dim), dtype=np.float32)

    dim = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP: {embeddings.shape[0]} vectors, dim={dim}")

    # IndexFlatIP: exact inner product search (cosine for normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(output_path))
    logger.info(f"FAISS index saved to {output_path} ({index.ntotal} vectors)")


# =============================================================================
# Step 6: BM25 index
# =============================================================================

def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 indexing.

    Lowercase + split on non-alphanumeric characters (preserves Unicode).

    Args:
        text: Raw text string.

    Returns:
        List of lowercase tokens.
    """
    text = text.lower()
    return re.findall(r"[^\W_]+", text, re.UNICODE)


def build_bm25_index(
    chunks: List[ChunkDict],
    output_path: Path | None = None,
) -> None:
    """Build and pickle a BM25Okapi index from chunk texts.

    Args:
        chunks: List of chunk dicts (each must have "text" key).
        output_path: Where to save the pickled index (default from config).
    """
    if output_path is None:
        output_path = config.bm25_index_path

    if not chunks:
        logger.warning("No chunks for BM25 index, skipping")
        return

    from rank_bm25 import BM25Okapi

    logger.info(f"Tokenizing {len(chunks)} chunks for BM25...")
    tokenized_corpus = [_tokenize_for_bm25(c["text"]) for c in tqdm(chunks, desc="BM25 tokenize")]

    logger.info("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized_corpus)

    with open(output_path, "wb") as f:
        pickle.dump(bm25, f)

    logger.info(f"BM25 index saved to {output_path} ({len(chunks)} documents)")


# =============================================================================
# Step 7: Export chunks to JSONL
# =============================================================================

def export_chunks_jsonl(
    chunks: List[ChunkDict],
    output_path: Path | None = None,
) -> None:
    """Write all chunks to a JSONL file (one JSON object per line).

    Args:
        chunks: List of chunk dicts.
        output_path: Where to save (default from config).
    """
    if output_path is None:
        output_path = config.chunk_jsonl_path

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(chunks)} chunks to {output_path}")


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(
    limit: int | None = None,
    skip_ocr: bool = False,
    output_dir: Path | None = None,
    embedding_model: str | None = None,
) -> None:
    """Run the full data preparation pipeline.

    Steps:
        1. Load MASTER_INDEX.csv, deduplicate to unique PDFs.
        2. For each PDF: extract text (searchable/OCR/mixed).
        3. Chunk all extracted texts.
        4. Generate embeddings for all chunks.
        5. Build and save FAISS index.
        6. Build and save BM25 index.
        7. Export chunks to chunks.jsonl.

    Args:
        limit: If set, process only the first N PDFs (for debugging).
        skip_ocr: If True, skip OCR for scanned PDFs (faster but incomplete).
        output_dir: Override output directory for all generated files.
    """
    # Override config paths if output_dir is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.chunk_jsonl_path = output_dir / "chunks.jsonl"
        config.faiss_index_path = output_dir / "faiss.index"
        config.bm25_index_path = output_dir / "bm25_index.pkl"

    # Reset chunk counter
    _reset_chunk_counter()

    # ---- Load master index ----
    df = load_master_index()
    unique_pdfs = get_unique_pdfs(df)

    if limit:
        unique_pdfs = unique_pdfs.head(limit)
        logger.info(f"Limited to {limit} PDFs")

    # ---- Extract text from all PDFs ----
    all_chunks: List[ChunkDict] = []
    success_count = 0
    fail_count = 0
    skipped_ocr = 0

    for _, row in tqdm(
        unique_pdfs.iterrows(),
        total=len(unique_pdfs),
        desc="Extracting PDFs",
    ):
        # Build full path: data_dir / entity / filename
        # The filename in CSV is like "contrats/acquired/precistec_xxx.pdf"
        pdf_path = config.data_dir / row["entity"] / row["filename"]

        format_type = row.get("format", "searchable")
        language = row.get("language", "en")

        if skip_ocr and format_type in ("scanned", "mixed"):
            skipped_ocr += 1
            continue

        text = extract_text_from_pdf(pdf_path, format_type, language)

        if not text:
            fail_count += 1
            logger.debug(f"  [FAIL] No text: {row['filename']}")
            continue

        success_count += 1

        # Build metadata for this PDF's chunks
        metadata = {
            "source_file": f"{row['entity']}/{row['filename']}",
            "entity": row.get("entity", "unknown"),
            "language": language,
            "format": format_type,
            "pages": int(row.get("pages", 0)) if pd.notna(row.get("pages")) else 0,
            "classification": str(row.get("classification", "")),
        }

        # Chunk the text
        chunks = chunk_text(text, metadata)
        all_chunks.extend(chunks)

    logger.info(
        f"PDF extraction complete: {success_count} success, "
        f"{fail_count} failed, {skipped_ocr} skipped (OCR disabled), "
        f"{len(all_chunks)} total chunks"
    )

    if not all_chunks:
        logger.error("No chunks generated! Aborting.")
        return

    # ---- Generate embeddings ----
    embeddings, all_chunks = generate_embeddings(all_chunks, model_name=embedding_model)

    # ---- Build FAISS index ----
    build_faiss_index(embeddings)

    # ---- Build BM25 index ----
    build_bm25_index(all_chunks)

    # ---- Export chunks ----
    export_chunks_jsonl(all_chunks)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete: {len(all_chunks)} chunks, "
                 f"{embeddings.shape[0]} embeddings, "
                 f"{success_count} PDFs processed.")
    logger.info(f"Output files:")
    logger.info(f"  {config.chunk_jsonl_path}")
    logger.info(f"  {config.faiss_index_path}")
    logger.info(f"  {config.bm25_index_path}")
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """Parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Data preparation pipeline: PDF → chunks → FAISS + BM25 indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python data_prepare.py                        # Full pipeline
    python data_prepare.py --skip-ocr             # Skip scanned PDFs (faster)
    python data_prepare.py --limit 50             # Process only 50 PDFs
    python data_prepare.py --output-dir ./indexes  # Custom output directory
    python data_prepare.py --verbose --limit 10   # Debug mode with 10 PDFs
        """,
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Process only the first N PDFs (for testing)",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR for scanned and mixed PDFs (faster but incomplete)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Directory for output files (chunks.jsonl, faiss.index, bm25_index.pkl)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override the embedding model name (default from config). "
             "Use a local path or a different HuggingFace model, e.g., "
             "'BAAI/bge-small-zh-v1.5' or 'all-MiniLM-L6-v2'.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_pipeline(
        limit=args.limit,
        skip_ocr=args.skip_ocr,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
