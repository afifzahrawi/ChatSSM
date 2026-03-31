"""
chatssm_app.py  -  ChatSSM Streamlit Application
=================================================
Legal Q&A system for Suruhanjaya Syarikat Malaysia (SSM).
Retrieves answers strictly from pre-processed legal documents.

Prerequisites
-------------
  1. Python 3.11  (avoid 3.14 - incompatible with pandas/numpy C extensions)
  2. Install deps:   pip install -r requirements.txt
  3. Pre-process:    python preprocess.py
  4. Run Ollama:     ollama serve
  5. Pull models:    ollama pull qwen3-embedding:8b
                     ollama pull qwen3:8b
                     ollama pull glm-ocr
  6. Start app:      streamlit run chatssm_app.py

Architecture
------------
  SourceRegistry   - reads knowledge_sources.json
  EmbeddingService - two-level cache (RAM + disk), numpy-normalised vectors
  DocumentIndex    - per-source numpy matrix; O(1) vectorised cosine search
  KnowledgeBase    - orchestrates all indexes; merges and ranks results
  FeedbackStore    - structured feedback JSON log
  PromptOptimizer  - analyzes failure patterns; dynamically patches system prompt
  LLMService       - strict-grounding prompt; direct Ollama REST call
  StorageService   - chat history JSON + Q&A CSV log
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pickle
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple
from filelock import FileLock
from rank_bm25 import BM25Okapi # type: ignore
from collections import OrderedDict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import html
import tempfile, shutil
import csv

from chunk import Chunk, SearchResult
from memory_manager import MemoryManager
from intent_form_agent import IntentFormAgent

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chatssm")

# =============================================================================
# CATEGORY REGISTRY  (fixed order controls sidebar display)
# =============================================================================

CATEGORIES: List[str] = [
    "Legislations",
    "Practice Notes",
    "Practice Directives",
    "Guidelines",
    "Circular",
    "FAQ",
    "Forms",
]

CATEGORY_ICONS: Dict[str, str] = {
    "Legislations":        "📜",
    "Practice Notes":      "📝",
    "Practice Directives": "📋",
    "Guidelines":          "📌",
    "Circular":            "🔔",
    "FAQ":                 "❓",
    "Forms":               "📄",
}

# =============================================================================
# CONFIGURATION
# =============================================================================


class AppConfig:
    # ── Ollama ────────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str   = "http://localhost:11434"
    EMBEDDING_MODEL: str   = "qwen3-embedding:8b"
    LLM_MODEL:       str   = "qwen3:8b"

    # ── LLM sampling (deterministic = no hallucination drift) ─────────────────
    LLM_TEMPERATURE: float = 0.0
    LLM_TOP_P:       float = 0.9
    LLM_TOP_K:       int   = 20
    LLM_MAX_TOKENS:  int   = 1500
    LLM_TIMEOUT:     int   = 300    # seconds
    LLM_NUM_CTX: int = 8192   #Ollama default is 2048; set higher to fit system prompt + long contexts without truncation. Must be >= LLM_MAX_TOKENS + max context chunk size.
    LLM_NUM_GPU:       int = int(os.environ.get("CHATSSM_LLM_NUM_GPU", "-1"))

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_TIMEOUT: int = 60     # seconds per call
    EMBEDDING_WORKERS: int = 4      # Parallel workers for index building.
    EMBEDDING_NUM_GPU: int = int(os.environ.get("CHATSSM_EMBEDDING_NUM_GPU", "-1"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.35
    TOP_K_PER_SOURCE:     int   = 4
    GLOBAL_TOP_K:         int   = 12   # total chunks sent to LLM
    MAX_CHUNK_CHARS:     int   = 1200  # Truncate chunks in the prompt to this many characters to avoid hitting LLM token limits.
    MAX_TABLE_CHARS:  int = 2500
    BM25_WEIGHT: float = 0.2  # 20% BM25, 80% vector — legal domain favors semantic

    # ── Paths ─────────────────────────────────────────────────────────────────
    SOURCES_CONFIG:  str = os.path.join("knowledge_base", "knowledge_sources.json")
    SOURCES_DIR:     str = os.path.join("knowledge_base", "sources")
    PROCESSED_DIR:   str = os.path.join("knowledge_base", "processed")
    CACHE_DIR:       str = os.path.join("knowledge_base", "embeddings")
    FORMS_FILE:      str = os.path.join("knowledge_base", "forms.json")
    DATA_DIR:        str = "qa_data"
    USER_DATA:       str = "user_data"

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in [cls.SOURCES_DIR, cls.PROCESSED_DIR, cls.CACHE_DIR, cls.DATA_DIR, cls.USER_DATA]:
            try:
                os.makedirs(d, exist_ok=True)
            except OSError as exc:
                logger.warning("Could not create directory '%s': %s", d, exc)


AppConfig.ensure_dirs()

_QA_LOG_FILE       = os.path.join(AppConfig.DATA_DIR, "qa_log.csv")
_FEEDBACK_FILE     = os.path.join(AppConfig.DATA_DIR, "feedback.json")
_EMBEDDING_CACHE   = os.path.join(AppConfig.CACHE_DIR, "embedding_cache.pkl")
_RESPONSE_CACHE_FILE = os.path.join(AppConfig.DATA_DIR, "response_cache.pkl")

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SourceEntry:
    key:             str
    name:            str
    category:        str
    type:            str              # "pdf" or "csv"
    enabled:         bool = True
    url:             Optional[str] = None
    local_path:      Optional[str] = None
    # BUG FIX: was missing from SourceEntry — getattr on this field always
    # returned [] making Act-based retrieval filtering completely non-functional.
    relates_to_acts: List[str] = field(default_factory=list)

    @property
    def processed_csv(self) -> str:
        """Path to the pre-processed CSV produced by preprocess.py."""
        return os.path.join(AppConfig.PROCESSED_DIR, f"{self.key}.csv")

    @property
    def is_ready(self) -> bool:
        """True when the pre-processed CSV exists on disk."""
        return os.path.exists(self.processed_csv)

# =============================================================================
# SOURCE REGISTRY  (folder scan + optional JSON overrides)
# =============================================================================

_VALID_CATEGORIES: List[str] = [
    "Legislations", "Practice Notes", "Practice Directives",
    "Guidelines", "Circular", "FAQ", "Forms",
]
_VALID_DOC_TYPES: List[str] = ["act", "general", "faq", "gazette", "slide", "others"]

# =============================================================================
# QUERY ANALYZER
# =============================================================================

_CONTEXT_REF_RE = re.compile(
    r'\b(it|its|this|that|these|those|they|them|their|'
    r'the\s+(?:company|director|secretary|fee|deadline|section|act|'
    r'provision|requirement|period|amount|penalty)|'
    r'same|above|mentioned|previous|last|earlier)\b',
    re.IGNORECASE,
    )

_CONTINUATION_RE = re.compile(
    r'^(?:what\s+about|how\s+about|and\s+(?:what|how)|also|'
    r'additionally|furthermore|can\s+it|does\s+it|is\s+it|'
    r'yes\b|yes\s+please|yes\s+i\s+would|yes\s+i\s+want|'
    r'sure\b|sure\s+please|okay\s+then|go\s+ahead|please\s+do|'
    r'tell\s+me\s+more|i\s+would\s+like\s+to\s+know|please\s+explain)\b',
    re.IGNORECASE,
)

_MALAY_MARKERS = re.compile(
    r'\b(apakah|berapakah|bagaimana|bilakah|siapakah|mengapa|adakah|'
    r'bolehkah|perlukah|syarikat|pengarah|pemegang\s+saham|pendaftaran|'
    r'penubuhan|setiausaha|yuran|daftar|akta|peraturan|dan|atau|untuk|'
    r'dengan|dalam|kepada|daripada|adalah|tidak|perlu|boleh|akan|telah|'
    r'sebuah|seorang|semua|setiap|syarat|bayaran|penyata|tahunan|'
    r'mesyuarat|resolusi|modal|saham|berhad|sdn\.?\s*bhd\.?|'
    r'terangkan|jelaskan|apakah|beritahu|nyatakan|huraikan|'   # question starters
    r'secara|talian|atas|bawah|cara|jenis|jika|kalau|'         # "secara atas talian" = online
    r'seksyen|fasal|subseksyen|peruntukan|jadual|lampiran|'    # legal structural terms
    r'ialah|iaitu|iaitu|bermaksud|merujuk|berkaitan|'          # definitional words
    r'perlu|mesti|wajib|hendaklah|haruslah|dilarang|'   # modal/obligation verbs
    r'juga|sahaja|hanya|selepas|sebelum|semasa|antara|'       # common function words
    r'korporat|pengurusan|lembaga|ahli|pengurus|'    # corporate vocabulary
    r'denda|penalti|kesalahan|hukuman|liabiliti|tanggungjawab|' # enforcement terms
    r'lesen|permit|kelulusan|kebenaran|perakuan|sijil|'     # approval terms
    r'mcm|mana|nak|tak|dah|lah|kan|pun|je|kot|bt|sbb|bisnes|bayar|kena|buat|macam|'
    r'yang|tentang|mengenai|ada|apa|bila|ini|itu|saya|anda|dia|kita|mereka)\b',
    re.IGNORECASE,
)

_QUERY_SYNONYMS: Dict[str, str] = {
    r'\bagm\b':  'annual general meeting',
    r'\beot\b':  'extension of time',
    r'\bcosec\b': 'company secretary',
    r'\begt\b':  'extraordinary general meeting',
    r'\begm\b':  'extraordinary general meeting',
}

_GREETING_RE = re.compile(
    r'^(?:hi+|hello+|hey+|good\s+(?:morning|afternoon|evening|day)|'
    r'hai|helo|selamat\s+(?:pagi|tengahari|petang|malam|datang)|'
    r'apa\s+khabar|assalamualaikum|salam)\W*$',
    re.IGNORECASE,
)

_GREETING_RESPONSES = {
    "en": (
        "Hello! I'm ChatSSM, your legal assistant for Malaysian company law. "
        "Feel free to ask me anything about the Companies Act, LLP Act, "
        "business registration, or any other SSM-related matter."
    ),
    "ms": (
        "Hai! Saya ChatSSM, pembantu undang-undang anda untuk undang-undang syarikat Malaysia. "
        "Sila tanya saya apa-apa sahaja berkaitan Akta Syarikat, pendaftaran perniagaan, "
        "atau sebarang perkara berkaitan SSM."
    ),
}

def _expand_query(text: str) -> str:
    """Expand common abbreviations before embedding to improve retrieval."""
    for pattern, replacement in _QUERY_SYNONYMS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def _detect_language(text: str) -> str:
    """
    Returns 'ms' (Malay), 'en' (English), or 'mixed'.

    Uses three signals in priority order:
    1. Malay-specific letter patterns that never appear in English
    2. High-frequency Malay function words (short, common, colloquial)
    3. Word-count ratio fallback
    """
    t = text.lower().strip()

    words = re.findall(r"[a-zA-Z']+", t)
    if not words:
        return "en"

    malay_hits = len(_MALAY_MARKERS.findall(t))
    ratio = malay_hits / len(words)

    # Lower threshold — catch short colloquial queries
    if ratio >= 0.25:
        return "ms"
    elif ratio >= 0.10:
        return "mixed"
    return "en"

def _filename_to_key(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    key  = stem.lower()
    key  = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


class SourceRegistry:
    """
    Discovers sources from two tracks and merges them:

    Track 1 - Folder scan  (knowledge_base/sources/<Category>/<doc_type>/)
        Drop a PDF in the right folder — it is automatically registered.
        The folder path determines category and doc_type.

    Track 2 - JSON overrides  (knowledge_sources.json, optional)
        Use only when you need to override a field, add a remote URL source,
        or explicitly disable a folder-discovered file.

    To add a new source: just drop the file in the right folder and click
    "Reload Sources" in the sidebar (or restart the app).
    """

    def __init__(self) -> None:
        self._sources: List[SourceEntry] = []
        self._load()

    # ── Discovery ─────────────────────────────────────────────────────────────

    def _scan_folders(self) -> Dict[str, SourceEntry]:
        discovered: Dict[str, SourceEntry] = {}
        sources_dir = AppConfig.SOURCES_DIR
        if not os.path.isdir(sources_dir):
            return discovered

        for cat_name in os.listdir(sources_dir):
            cat_path = os.path.join(sources_dir, cat_name)
            if not os.path.isdir(cat_path) or cat_name not in _VALID_CATEGORIES:
                continue

            for dt_name in os.listdir(cat_path):
                dt_path = os.path.join(cat_path, dt_name)
                if not os.path.isdir(dt_path) or dt_name not in _VALID_DOC_TYPES:
                    continue

                for fname in sorted(os.listdir(dt_path)):
                    _SUPPORTED = (".pdf", ".csv", ".png", ".jpg", ".jpeg", ".webp", ".pptx", ".pptm")
                    if not fname.lower().endswith(_SUPPORTED):
                        continue
                    fpath = os.path.join(dt_path, fname)
                    key   = _filename_to_key(fname)
                    if key in discovered:
                        logger.warning(
                            "KEY COLLISION: '%s' and '%s' both map to key '%s'. "
                            "Second file will not be processed. Rename one to resolve.",
                            discovered[key].local_path, fpath, key
                        )
                        continue
                    
                    ext = os.path.splitext(fname)[1].lower()
                    ftype = "csv" if ext == ".csv" else ("pptx" if ext in (".pptx", ".pptm") else ("image" if ext in (".png", ".jpg", ".jpeg", ".webp") else "pdf"))
                    
                    discovered[key] = SourceEntry(
                        key        = key,
                        name       = os.path.splitext(fname)[0],
                        category   = cat_name,
                        type       = ftype,
                        enabled    = True,
                        url        = None,
                        local_path = fpath,
                    )
        return discovered

    def _load_json(self) -> Dict[str, SourceEntry]:
        path = AppConfig.SOURCES_CONFIG
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return {}

        overrides: Dict[str, SourceEntry] = {}
        _VALID_TYPES = {"pdf", "csv", "image", "pptx"}
        for raw in data.get("sources", []):
            if "_note" in raw or "_comment" in raw:
                continue
            raw_key = raw.get("key")
            if not raw_key:
                logger.warning("Skipping JSON entry with missing 'key': %s", raw)
                continue
            se_type = raw.get("type", "pdf")
            if se_type not in _VALID_TYPES:
                logger.warning("Unknown source type '%s' for key '%s'. Defaulting to 'pdf'.", se_type, raw_key)
                se_type = "pdf"
            try:
                overrides[raw_key] = SourceEntry(
                    key             = raw_key,
                    name            = raw.get("name", raw_key),
                    category        = raw.get("category", ""),
                    type            = se_type,
                    enabled         = raw.get("enabled", True),
                    url             = raw.get("url"),
                    local_path      = raw.get("local_path"),
                    relates_to_acts = raw.get("relates_to_acts", []),
                )
            except Exception as exc:
                logger.warning("Skipping malformed JSON entry '%s': %s", raw_key, exc)
        return overrides

    def _load(self) -> None:
        folder  = self._scan_folders()
        json_ov = self._load_json()

        merged: Dict[str, SourceEntry] = dict(folder)

        for key, je in json_ov.items():
            if key in merged:
                base = merged[key]
                merged[key] = SourceEntry(
                    key             = key,
                    name            = je.name if je.name != key else base.name,
                    category        = je.category or base.category,
                    type            = je.type,
                    enabled         = je.enabled,
                    url             = je.url or base.url,
                    local_path      = je.local_path or base.local_path,
                    relates_to_acts = je.relates_to_acts or base.relates_to_acts,
                )
            else:
                merged[key] = je

        self._sources = sorted(
            merged.values(), key=lambda s: (s.category, s.name)
        )
        enabled = sum(1 for s in self._sources if s.enabled)
        logger.info(
            "Loaded %d sources (%d enabled): %d from folders, %d from JSON.",
            len(self._sources), enabled, len(folder), len(json_ov),
        )

    def reload(self) -> None:
        self._sources.clear()
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def all_enabled(self) -> List[SourceEntry]:
        return [s for s in self._sources if s.enabled]

    def by_category(self) -> Dict[str, List[SourceEntry]]:
        grouped: Dict[str, List[SourceEntry]] = {c: [] for c in CATEGORIES}
        for s in self.all_enabled():
            grouped.setdefault(s.category, []).append(s)
        return grouped

    def get(self, key: str) -> Optional[SourceEntry]:
        return next((s for s in self._sources if s.key == key), None)

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.all_enabled():
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts

# =============================================================================
# FORM REGISTRY
# =============================================================================

@dataclass
class FormEntry:
    form_id:          str
    form_number:      str
    name:             str
    links:            List[Dict]
    related_sections: List[str]
    resource_type:    str = "form" 
    _embedding:       Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def primary_url(self) -> str:
        if self.links:
            return self.links[0].get("url", "")
        return ""

# FormRegistry and its sentinel are retained for legacy compatibility but
# form detection is now handled entirely by IntentFormAgent.  The registry
# is no longer called from main().
_ASK_USER_SENTINEL = object()   # kept so old pickled session_state doesn't crash

class FormRegistry:
    """
    Loads SSM forms from forms.json and supports two retrieval modes:
      1. Explicit — form number/keyword detected in query
      2. Semantic  — embedding similarity against form purpose
    """

    # Explicit form request patterns
    _EXPLICIT_RE = re.compile(
        r'\b(?:form|borang)\s*(?:no\.?\s*)?([0-9]+[A-Za-z]?|[A-Za-z][0-9]*)\b'
        r'|\b(?:give|get|show|download|need|nak|minta|dapatkan)\s+(?:me\s+)?'
        r'(?:the\s+)?(?:form|borang)\b',
        re.IGNORECASE,
    )
    # Implicit intent patterns — procedural questions that usually require a form
    _IMPLICIT_RE = re.compile(
        r'\b(?:how\s+(?:to|do\s+I)|cara|nak|steps?\s+to|procedure\s+for|'
        r'process\s+(?:to|for)|langkah|apply\s+for|daftar|register|submit|'
        r'lodge|file|kemukakan|permohonan)\b',
        re.IGNORECASE,
    )

    _ACTION_RE = re.compile(
        r'\b(?:'
        r'how\s+(?:to|do\s+I|can\s+I)\b'
        r'|where\s+(?:to|do\s+I|can\s+I|should\s+I)\b'        
        r'|cara\b'
        r'|nak\s+(?:daftar|hantar|kemukakan|tukar|ubah|mohon|laksana)\b'
        r'|steps?\s+to\b|procedure\s+(?:to|for)\b'
        r'|(?:apply|applying)\s+for\b'
        r'|register\s+(?:a|my|our|as|with|for)\b'              
        r'|registering\s+(?:a|my|our|as|with|for)\b'
        r'|(?:submit|lodge|file|convert|change)\s+(?:a|the|my|an)\b'
        r'|langkah[-\s]langkah\b|tatacara\b'
        r'|(?:tukar|daftar|kemukakan|mohon|hantar)\s+\w+'
        r'|di\s+mana\s+(?:nak|untuk|boleh)\b'                  
        r')',
        re.IGNORECASE,
    )
    _INFORMATIONAL_RE = re.compile(
        r'\b(?:what\s+(?:is|are|happens?|if|does|did|do)|'
        r'what\'s\b|'
        r'tell\s+me\s+(?:about|what|how)\b|'
        r'explain\s+(?:section|the|what|how|why)\b|'
        r'describe\b|'
        r'penalty|penalti|denda|offence|kesalahan|'
        r'requirement|keperluan|syarat|'
        r'difference|perbezaan|beza|'
        r'explain|terangkan|jelaskan|'
        r'why|mengapa|sebab)\b',
        re.IGNORECASE,
    )

    _FORM_REQUEST_RE = re.compile(
        r'\b(?:'
        r'give\s+me\s+(?:the\s+)?(?:form|borang)'
        r'|show\s+(?:me\s+)?(?:the\s+)?(?:form|borang)'
        r'|what\s+form\s+(?:do\s+I|should\s+I|is\s+needed)'
        r'|(?:provide|send|get)\s+(?:me\s+)?(?:the\s+)?(?:form|borang)'
        r'|(?:minta|dapatkan|tunjuk)\s+(?:borang|form)'
        r'|borang\s+(?:apa|mana|yang)'
        r')',
        re.IGNORECASE,
    )

    MAX_FORMS_EXPLICIT = 3   # hard cap for explicit form number requests
    MAX_FORMS_IMPLICIT = 4   # hard cap for context-derived matching; dynamic logic may return fewer
    SEMANTIC_THRESHOLD = 0.60

    def __init__(self, emb: "EmbeddingService") -> None:
        self._emb   = emb
        self._forms: List[FormEntry] = []
        self._load()

    def _load(self) -> None:
        path = AppConfig.FORMS_FILE
        if not os.path.exists(path):
            logger.warning("forms.json not found at %s — form suggestions disabled.", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for d in data:
                # Backward-compat: if old record has "url" but no "links",
                # promote it to a single-item links list with type "pdf"
                if "url" in d and "links" not in d:
                    d["links"] = [{"type": "pdf", "url": d.pop("url")}]
                elif "url" in d:
                    d.pop("url")   # drop redundant old field if links already present
                self._forms.append(FormEntry(**{
                    k: v for k, v in d.items()
                    if k in FormEntry.__dataclass_fields__
                }))
            logger.info("FormRegistry: loaded %d form(s).", len(self._forms))
        except Exception as exc:
            logger.error("FormRegistry: failed to load forms.json: %s", exc)

    def detect(
        self,
        query: str,
        search_query: str = "",
        result_chunks: Optional[List[str]] = None,
        last_sections: Optional[Dict[tuple, int]] = None,
    ) -> list:
        """
        Strict section-first form retrieval.

        Returns only forms whose linked sections are the DOMINANT sections
        in the retrieved context.  Never guesses or falls back to generic forms.

        Two bugs fixed vs previous version:
          1. _SEC_RE had a \\b that prevented subsection capture — '41(1)' was
             parsed as '41' with no child, making subsection discrimination impossible.
          2. All retrieved sections were weighted equally, so incidental sections
             (e.g. Section 14 mentioned once in a 12-chunk answer about Section 40)
             competed on equal footing with the dominant section.

        Scoring now weights each section match by the number of chunks it appears
        in.  A section mentioned in 3 of 12 chunks outscores one mentioned once.
        """

        if not self._forms:
            return []

        # ── Corrected section extraction regex ────────────────────────────────
        # The original pattern r'\bsection\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?\b'
        # had \\b BETWEEN groups which matched before '(' in '41(1)' — causing
        # the subsection to never be captured.  Fixed by replacing terminal \\b
        # with a lookahead that works after optional ')'.
        _SEC_RE = re.compile(
            r'\b(?:section|seksyen)\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?'
            r'(?=\b|\s|\.|,|;|\)|\(|$)',
            re.IGNORECASE,
        )

        # ── Gate 0: Explicit section reference in user query ──────────────────
        # Only enter this path if the query is NOT purely informational.
        # A query like "what is section 40" mentions a section but does not
        # require a form — the user wants to understand the law, not file anything.
        # Allow through if: (a) no informational signal, OR (b) action intent present.
        _is_informational_only = (
            self._INFORMATIONAL_RE.search(query) is not None
            and self._ACTION_RE.search(query) is None
        )
        if not _is_informational_only:
            query_section_refs: set = set()
            for m in _SEC_RE.finditer(query):
                query_section_refs.add((m.group(1).lower(), m.group(2).lower() if m.group(2) else None))

            if query_section_refs:
                scored = []
                for form in self._forms:
                    sec_entries = [
                        rs for rs in form.related_sections
                        if not rs.lower().startswith("companies act")
                        and not rs.lower().startswith("llp act")
                    ]
                    score = sum(self._score_ref(rs, query_section_refs) for rs in sec_entries)
                    if score >= 2 or (score == 1 and len(sec_entries) == 1):
                        scored.append((score, form))
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return self._pdf_only([f for _, f in scored[:self.MAX_FORMS_EXPLICIT]])
                # No form matches the explicitly cited section — return nothing
                return []

        # ── Gate 1: Explicit form number request ──────────────────────────────
        if self._EXPLICIT_RE.search(query):
            num_match = re.search(
                r'\b(?:form|borang)\s*(?:no\.?\s*)?([0-9]+[A-Za-z]?)',
                query, re.IGNORECASE,
            )
            if num_match:
                target = num_match.group(1).strip().lower()
                exact = [
                    f for f in self._forms
                    if f.form_number and target in f.form_number.lower()
                ]
                if exact:
                    return self._pdf_only(exact[:self.MAX_FORMS_EXPLICIT])
                
        # ── Gate 1.5: Follow-up form request — reuse previous section context ─────
        if self._FORM_REQUEST_RE.search(query) and last_sections:
            # User is explicitly asking for a form from the previous procedure.
            # Skip all action/informational gates and go straight to scoring
            # using the sections we already know are relevant.
            _prev_refs = set(last_sections.keys())
            _prev_max  = max(last_sections.values())

            def _prev_weighted(rs_string: str) -> float:
                m = re.match(
                    r'(?:section|seksyen)\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?',
                    rs_string.strip(), re.IGNORECASE,
                )
                if not m:
                    return 0.0
                f_parent = m.group(1).lower()
                f_child  = m.group(2).lower() if m.group(2) else None
                for r_parent, r_child in _prev_refs:
                    if r_parent != f_parent:
                        continue
                    freq = last_sections[(r_parent, r_child)]
                    if f_child is not None and r_child is not None:
                        if f_child == r_child:
                            return 2.0 * freq
                        continue
                    return 1.0 * freq
                return 0.0

            prev_scored = []
            for form in self._forms:
                sec_entries = [
                    rs for rs in form.related_sections
                    if not rs.lower().startswith("companies act")
                    and not rs.lower().startswith("llp act")
                ]
                total = sum(_prev_weighted(rs) for rs in sec_entries)
                if total > 0:
                    prev_scored.append((total, form))

            if prev_scored:
                prev_scored.sort(key=lambda x: x[0], reverse=True)
                return self._pdf_only([f for _, f in prev_scored[:self.MAX_FORMS_EXPLICIT]])
            # Previous context found but no form matches — return nothing
            return []

        # ── Gate 2a: Action intent required ───────────────────────────────────
        check_text = query + " " + search_query
        if not self._ACTION_RE.search(check_text):
            return []

        # ── Gate 2b: Block pure informational queries ─────────────────────────
        if self._INFORMATIONAL_RE.search(query):
            return []

        # ── Gate 3: Frequency-weighted section matching ────────────────────────
        # Build a frequency map: how many individual chunks mention each section.
        # Sections appearing in more chunks are more central to the answer.
        if not result_chunks:
            return []

        section_freq: Dict[tuple, int] = {}
        for chunk in result_chunks:
            chunk_refs: set = set()
            for m in _SEC_RE.finditer(chunk):
                chunk_refs.add((m.group(1).lower(), m.group(2).lower() if m.group(2) else None))
            for ref in chunk_refs:
                section_freq[ref] = section_freq.get(ref, 0) + 1

        if not section_freq:
            # Procedural intent detected but no section refs in chunks — return nothing
            return []

        all_refs = set(section_freq.keys())
        # Dominant frequency: how many chunks does the most-cited section appear in
        max_freq = max(section_freq.values())
        distinct_parents = len({r[0] for r in all_refs})
        MIN_FORM_FREQ = 2 

        def _weighted_score(rs_string: str) -> float:
            m = re.match(
                r'(?:section|seksyen)\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?',
                rs_string.strip(), re.IGNORECASE,
            )
            if not m:
                return 0.0
            f_parent = m.group(1).lower()
            f_child  = m.group(2).lower() if m.group(2) else None

            best = 0.0
            for r_parent, r_child in all_refs:
                if r_parent != f_parent:
                    continue
                freq = section_freq[(r_parent, r_child)]
                if f_child is not None and r_child is not None:
                    if f_child == r_child:
                        base = 2.0                    # exact subsection match
                    else:
                        continue                      # different subsection — skip
                else:
                    base = 1.0                        # parent-level match
                best = max(best, base * freq)
            return best

        scored = []
        for form in self._forms:
            sec_entries = [
                rs for rs in form.related_sections
                if not rs.lower().startswith("companies act")
                and not rs.lower().startswith("llp act")
            ]
            if not sec_entries:
                continue

            total = sum(_weighted_score(rs) for rs in sec_entries)
            if total == 0:
                continue

            # Require that this form's section is sufficiently dominant.
            # "Dominant" = the section appears in at least HALF as many chunks
            # as the most-cited section.  This blocks forms linked to incidental
            # sections (mentioned once) when the answer is primarily about a
            # different section (mentioned multiple times).
            form_max_freq = 0.0
            for rs in sec_entries:
                m2 = re.match(
                    r'(?:section|seksyen)\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?',
                    rs.strip(), re.IGNORECASE,
                )
                if not m2:
                    continue
                fp, fc = m2.group(1).lower(), (m2.group(2).lower() if m2.group(2) else None)
                for ref in all_refs:
                    if ref[0] == fp:
                        form_max_freq = max(form_max_freq, section_freq[ref])

            # Gate: form's section must appear in ≥ 50% of max_freq chunks
            if distinct_parents > 1 and (
                form_max_freq < max_freq * 0.6    # raised from 0.5 to 0.6
                or form_max_freq < MIN_FORM_FREQ  # must appear in ≥2 chunks regardless
            ):
                logger.debug(
                    "FormRegistry: skipping '%s' — section freq %.0f < 60%% of max %.0f or < min %d",
                    form.name[:40], form_max_freq, max_freq, MIN_FORM_FREQ,
                )
                continue

            scored.append((total, form))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Dynamic form limit:
        #   • Single dominant section → max 2 (one procedure, avoid overload)
        #   • Multiple dominant sections → up to 4 (complex workflow may need several)
        #   • Hard cap at MAX_FORMS_IMPLICIT to prevent overwhelming the user
        n_dominant = sum(1 for ref in all_refs if section_freq[ref] >= max_freq * 0.5)
        dynamic_limit = min(max(2, n_dominant * 2), self.MAX_FORMS_IMPLICIT)
        return self._pdf_only([f for _, f in scored[:dynamic_limit]])

    @staticmethod
    def _score_ref(rs_string: str, ref_pool: set) -> int:
        """Score one form section string against a pool of (parent, child) refs."""
        m = re.match(
            r'(?:section|seksyen)\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?',
            rs_string.strip(), re.IGNORECASE,
        )
        if not m:
            return 0
        f_parent = m.group(1).lower()
        f_child  = m.group(2).lower() if m.group(2) else None
        for r_parent, r_child in ref_pool:
            if r_parent != f_parent:
                continue
            if f_child is not None and r_child is not None:
                return 2 if f_child == r_child else 0
            return 1
        return 0

    @staticmethod
    def _pdf_only(forms: List["FormEntry"]) -> List["FormEntry"]:
        """Keep forms that have at least one usable link (pdf or portal)."""
        _VALID = {"pdf", "portal", "platform"}
        return [
            f for f in forms
            if any(l.get("type") in _VALID for l in f.links)
        ]

    def reload(self) -> None:
        self._forms.clear()
        self._load()


@st.cache_resource
def _form_registry() -> FormRegistry:
    return FormRegistry(_emb())

# =============================================================================
# EMBEDDING SERVICE  (RAM cache → disk cache → Ollama API)
# =============================================================================


def _bump(hit: bool) -> None:
    try:
        s = st.session_state.get("cache_stats", {"hits": 0, "misses": 0})
        s["hits" if hit else "misses"] += 1
        st.session_state["cache_stats"] = s
    except Exception:
        pass 

class EmbeddingService:
    """
    Embeddings are unit-normalised so cosine_similarity == dot_product.

    Two-level cache:
      L1 - in-process dict  (instant, lives for the current server process)
      L2 - pickle on disk   (survives app restarts)

    Thread safety:
      _disk_lock  - serialises all writes AND reads to the pickle dict
      _mem_lock   - serialises reads/writes to the shared in-memory dict
    """

    def __init__(self) -> None:
        self._mem:       Dict[str, np.ndarray] = {}
        self._disk:      Dict[str, np.ndarray] = self._load_disk()
        self._mem_lock:  threading.Lock        = threading.Lock()
        self._disk_lock: threading.Lock        = threading.Lock()

    # ── Public ────────────────────────────────────────────────────────────────

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text. Thread-safe."""
        if not text or not text.strip():
            return None

        key = hashlib.md5(text.encode()).hexdigest()

        with self._mem_lock:
            if key in self._mem:
                _bump(hit=True)
                return self._mem[key]

        # BUG FIX: check _disk under its own lock, not _mem_lock
        with self._disk_lock:
            if key in self._disk:
                _bump(hit=True)
                vec = self._disk[key]
                with self._mem_lock:
                    self._mem[key] = vec
                return vec

        _bump(hit=False)
        raw = self._call_api(text)
        if raw is None:
            return None

        vec = self._normalise(np.array(raw, dtype=np.float32))
        self._store(key, vec)
        return vec

    def embed_batch(
        self,
        texts: List[str],
        workers: int = AppConfig.EMBEDDING_WORKERS,
        progress_cb=None,
    ) -> List[Optional[np.ndarray]]:
        """
        Embed a list of texts in parallel.

        Arguments:
            texts       - list of strings to embed
            workers     - number of concurrent Ollama requests (default from AppConfig)
            progress_cb - optional callable(completed, total) called after each result

        Returns a list the same length as `texts`; failed embeddings are None.
        """
        total   = len(texts)
        results: Dict[int, Optional[np.ndarray]] = {}

        # Check cache first — avoid sending already-cached texts to Ollama at all
        to_fetch: Dict[int, str] = {}
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = None
                continue
            key = hashlib.md5(text.encode()).hexdigest()

            cached = False
            with self._mem_lock:
                if key in self._mem:
                    _bump(hit=True)
                    results[i] = self._mem[key]
                    cached = True

            if not cached:
                # BUG FIX: check _disk under _disk_lock, not _mem_lock
                with self._disk_lock:
                    if key in self._disk:
                        _bump(hit=True)
                        vec = self._disk[key]
                        with self._mem_lock:
                            self._mem[key] = vec
                        results[i] = vec
                        cached = True

            if not cached:
                to_fetch[i] = text

        logger.info(
            "embed_batch: %d cached, %d to fetch (workers=%d)",
            total - len(to_fetch), len(to_fetch), workers,
        )

        if not to_fetch:
            if progress_cb:
                progress_cb(total, total)
            return [results.get(i) for i in range(total)]

        # Fetch uncached texts in parallel
        completed = total - len(to_fetch)

        def _fetch(idx_text: Tuple[int, str]) -> Tuple[int, Optional[np.ndarray]]:
            idx, text = idx_text
            raw = self._call_api(text)
            if raw is None:
                return idx, None
            vec = self._normalise(np.array(raw, dtype=np.float32))
            k = hashlib.md5(text.encode()).hexdigest()
            self._store(k, vec)
            return idx, vec

        misses = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch, item): item[0] for item in to_fetch.items()}
            for future in as_completed(futures):
                try:
                    idx, vec = future.result()
                except Exception as exc:
                    logger.error("Embedding worker failed: %s", exc)
                    idx = futures[future]
                    vec = None
                results[idx] = vec
                completed += 1
                if vec is None:
                    misses += 1
                if progress_cb:
                    progress_cb(completed, total)

        for _ in range(misses):
            _bump(hit=False)

        self._save_disk()

        return [results.get(i) for i in range(total)]

    def clear_memory(self) -> None:
        with self._mem_lock:
            self._mem.clear()

    def clear_disk(self) -> None:
        with self._disk_lock:
            self._disk.clear()
        if os.path.exists(_EMBEDDING_CACHE):
            os.remove(_EMBEDDING_CACHE)
        logger.info("Embedding disk cache cleared.")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _store(self, key: str, vec: np.ndarray) -> None:
        """Thread-safe write to both cache levels. Does NOT flush disk."""
        with self._mem_lock:
            self._mem[key] = vec
        with self._disk_lock:
            self._disk[key] = vec

    def _call_api(self, text: str) -> Optional[List[float]]:
        """Single blocking HTTP call to Ollama. Safe to call from many threads."""
        try:
            r = requests.post(
                f"{AppConfig.OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model":   AppConfig.EMBEDDING_MODEL,
                    "prompt":  text,
                    "options": {"num_gpu": AppConfig.EMBEDDING_NUM_GPU},
                },
                timeout=AppConfig.EMBEDDING_TIMEOUT,
            )
            if r.status_code == 200:
                emb = r.json().get("embedding")
                if emb is None:
                    logger.warning("Ollama returned HTTP 200 but no 'embedding' key. Model may not be loaded.")
                return emb
            
            status = r.status_code
            if status == 404:
                logger.error(
                    "Embedding model '%s' not found (HTTP 404). "
                    "Fix: ollama pull %s",
                    AppConfig.EMBEDDING_MODEL, AppConfig.EMBEDDING_MODEL,
                )
            elif status == 429:
                logger.warning(
                    "Ollama rate-limited (HTTP 429). "
                    "Fix: reduce EMBEDDING_WORKERS in AppConfig (currently %d).",
                    AppConfig.EMBEDDING_WORKERS,
                )
            elif status == 503:
                logger.warning(
                    "Ollama unavailable (HTTP 503) — model may still be loading. "
                    "Fix: wait and retry, or check 'ollama ps'.",
                )
            else:
                logger.warning(
                    "Embeddings API returned HTTP %s. Response: %s",
                    status, r.text[:300],
                )
        except requests.ConnectionError:
            logger.error(
                "Cannot reach Ollama at %s. Is 'ollama serve' running?",
                AppConfig.OLLAMA_BASE_URL,
            )
        except requests.Timeout:
            logger.error(
                "Embedding request timed out after %ds (text length: %d chars). "
                "Fix: increase EMBEDDING_TIMEOUT or shorten chunks.",
                AppConfig.EMBEDDING_TIMEOUT, len(text),
            )
        except Exception as exc:
            logger.error("Embedding error: %s", exc)
        return None

    def _load_disk(self) -> Dict[str, np.ndarray]:
        if not os.path.exists(_EMBEDDING_CACHE):
            return {}
        try:
            with open(_EMBEDDING_CACHE, "rb") as fh:
                cache = pickle.load(fh)
            logger.info("Loaded %d cached embeddings from disk.", len(cache))
            return cache
        except Exception as exc:
            logger.warning("Disk cache load failed: %s", exc)
            if "numpy" in str(exc).lower():
                logger.info("Numpy version mismatch detected. Removing old cache.")
                try:
                    os.remove(_EMBEDDING_CACHE)
                except Exception:
                    pass
            return {}

    def _save_disk(self) -> None:
        """Thread-safe atomic disk flush."""
        with self._disk_lock:
            dir_ = os.path.dirname(_EMBEDDING_CACHE)
            tmp_path = None
            try:
                # Write to temp file
                fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
                try:
                    with os.fdopen(fd, "wb") as fh:
                        pickle.dump(self._disk, fh)
                    # fd is now fully closed — safe to move on Windows
                except Exception:
                    os.close(fd)   # ensure fd closed even if fdopen fails
                    raise
                try:
                    os.chmod(tmp_path, 0o600)
                except OSError:
                    pass           # chmod not supported on all Windows configs
                shutil.move(tmp_path, _EMBEDDING_CACHE)
                tmp_path = None    # move succeeded — no cleanup needed
            except Exception as exc:
                logger.warning("Disk cache save failed: %s", exc)
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

# =============================================================================
# DOCUMENT INDEX  (per-source numpy matrix store)
# =============================================================================

class DocumentIndex:
    """
    Stores all chunks for ONE source as a (N, D) numpy matrix.

    Search is a single matrix multiply: scores = matrix @ query_vec
    Embeddings are unit-normalised so dot product equals cosine similarity.
    """

    def __init__(
        self,
        source_key:  str,
        source_name: str,
        category:    str,
        emb:         EmbeddingService,
        registry:    SourceRegistry,
    ) -> None:
        self._key      = source_key
        self._name     = source_name
        self._cat      = category
        self._emb      = emb
        self._registry = registry
        self._chunks:  List[Chunk]          = []
        self._matrix:  Optional[np.ndarray] = None   # shape (N, D)
        self._bm25:    Optional[object]     = None

    # ── Readiness ─────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._matrix is not None and bool(self._chunks) and self._bm25 is not None

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> bool:
        logger.info(
            "Building index '%s' [%s] — %d rows.", self._name, self._cat, len(df)
        )

        raw: List[Chunk] = []
        for _, row in df.iterrows():
            content = str(row.get("content", "")).strip()
            if len(content) < 40:
                continue

            section = str(row.get("section", "")).strip()
            title   = str(row.get("section_title", "")).strip()
            part    = str(row.get("part", "")).strip()

            header = f"{section}"
            if part:
                header += f" ({part})"
            if title:
                header += f": {title}"
            chunk_text = f"{header}\n\n{content}" if header.strip(": ") else content

            src_entry = self._registry.get(self._key)
            chunk_type = (
                "visual" if "[VISUAL DESCRIPTION]" in chunk_text else
                "table"  if "[TABLE]"               in chunk_text else
                "text"
            )
            raw.append(Chunk(
                text             = chunk_text,
                source_key       = self._key,
                source_name      = self._name,
                category         = self._cat,
                section          = section,
                part             = part,
                relates_to_acts  = src_entry.relates_to_acts if src_entry else [],
                chunk_type       = chunk_type,
            ))

        return self._embed_and_build(raw)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, query_text: str, lang: str = "en") -> List[SearchResult]:
        # Raw cosine similarity — embeddings are unit-normalised so this is in [0, 1].
        # Do NOT normalise by per-source max: that inflates every source's best chunk
        # to 1.0 regardless of true relevance. With 83 sources all scoring ~1.0, the
        # global top-8 merge becomes noise — irrelevant chunks crowd out good ones.
        vector_scores = self._matrix @ query_vec

        # BM25 is relative within a source, not absolute — normalise per source so
        # it blends correctly with the absolute cosine scale.
        # Use re.findall (same tokenizer as index build) so punctuation is stripped:
        # "business?" → ["business"] not "business?" which has zero BM25 match.
        tokens = re.findall(r'[a-z0-9]+', query_text.lower())
        bm25_raw = np.array(self._bm25.get_scores(tokens), dtype=np.float32) # type: ignore
        b_max = bm25_raw.max()
        if b_max > 0:
            bm25_raw = bm25_raw / b_max

        w = 0.0 if lang == "ms" else AppConfig.BM25_WEIGHT
        scores = (1 - w) * vector_scores + w * bm25_raw

        k   = min(AppConfig.TOP_K_PER_SOURCE * 3, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        results: List[SearchResult] = []
        for idx in top:
            s = float(scores[idx])
            if s < AppConfig.SIMILARITY_THRESHOLD:
                break
            results.append(SearchResult(chunk=self._chunks[idx], score=s))
            if len(results) >= AppConfig.TOP_K_PER_SOURCE:
                break

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    @property
    def _cache_path(self) -> str:
        return os.path.join(AppConfig.CACHE_DIR, f"{self._key}_index.pkl")

    def save(self) -> None:
        tmp = None
        try:
            tmp = self._cache_path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump({"version": 3, "chunks": self._chunks, "matrix": self._matrix, "bm25": self._bm25}, fh)
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            shutil.move(tmp, self._cache_path)
            tmp = None
            logger.info("Saved index '%s'.", self._key)
        except Exception as exc:
            logger.warning("Could not save index '%s': %s", self._key, exc)
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def load(self) -> bool:
        if not os.path.exists(self._cache_path):
            return False
        try:
            with open(self._cache_path, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") != 3:
                logger.warning("Index '%s' version mismatch. Rebuilding.", self._key)
                self.delete_cache()
                return False
            
            self._chunks = payload["chunks"]
            self._matrix = payload["matrix"]

            if "bm25" not in payload:
                logger.warning("Index '%s' has no BM25 data (old cache). Rebuilding.", self._key)
                self.delete_cache()
                return False
            self._bm25 = payload["bm25"]

            if self._matrix is None or len(self._chunks) != self._matrix.shape[0]:
                logger.error(
                    "Index '%s' corrupted: %d chunks vs %d matrix rows. Deleting.",
                    self._key, len(self._chunks), 
                    self._matrix.shape[0] if self._matrix is not None else 0
                )
                self.delete_cache()
                return False
            logger.info("Loaded index '%s': %d chunks.", self._key, len(self._chunks))
            return True
        except Exception as exc:
            logger.warning("Failed to load index '%s': %s", self._key, exc)
            return False

    def delete_cache(self) -> None:
        if os.path.exists(self._cache_path):
            os.remove(self._cache_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _embed_and_build(self, raw: List[Chunk]) -> bool:
        total = len(raw)
        logger.info("  Embedding %d chunks (workers=%d) …", total, AppConfig.EMBEDDING_WORKERS)

        last_pct = [0]

        def _progress(done: int, tot: int) -> None:
            pct = int(done / tot * 100)
            if pct >= last_pct[0] + 10:
                logger.info("    %d%% (%d / %d)", pct, done, tot)
                last_pct[0] = pct

        MAX_EMBED_CHARS = 2000
        texts = [chunk.text[:MAX_EMBED_CHARS] for chunk in raw]
        vecs  = self._emb.embed_batch(texts, progress_cb=_progress)

        valid: List[Tuple[Chunk, np.ndarray]] = []
        for chunk, vec in zip(raw, vecs):
            if vec is not None:
                chunk.embedding = vec
                valid.append((chunk, vec))

        if not valid:
            logger.error("No embeddings produced for '%s'.", self._key)
            return False

        chunks_tuple, vs = zip(*valid)
        self._chunks = list(chunks_tuple)
        self._matrix = np.vstack(vs)
        tokenized = tokenized = [re.findall(r'[a-z0-9]+', chunk.text.lower()) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

        logger.info(
            "  Index ready: %d / %d chunks embedded for '%s'.",
            len(self._chunks), total, self._name,
        )
        return True


# =============================================================================
# KNOWLEDGE BASE  (orchestrator)
# =============================================================================


class KnowledgeBase:
    """
    One DocumentIndex per source.
    On first query, loads the pre-processed CSV and builds the index (cached).
    """

    def __init__(self, registry: SourceRegistry, emb: EmbeddingService) -> None:
        self._registry = registry
        self._emb      = emb
        self._indexes: Dict[str, DocumentIndex] = {}
        self._build_locks: Dict[str, threading.Lock] = {}

    # ── Index lifecycle ───────────────────────────────────────────────────────


    def get_or_build(self, source):
        """Return a ready index, building it if needed."""
        lock = self._build_locks.setdefault(source.key, threading.Lock())
        with lock:
            if source.key in self._indexes and self._indexes[source.key].is_ready():
                return self._indexes[source.key]

            idx = DocumentIndex(source.key, source.name, source.category, self._emb, self._registry)

            if idx.load():
                self._indexes[source.key] = idx
                return idx

            if not source.is_ready:
                logger.warning(
                    "Source '%s' has no processed CSV. Run:  python preprocess.py --key %s",
                    source.key, source.key,
                )
                return None

            try:
                df = pd.read_csv(source.processed_csv, encoding="utf-8")
            except Exception as exc:
                logger.error("Cannot read processed CSV for '%s': %s", source.key, exc)
                return None

            ok = idx.build(df)
            if ok:
                idx.save()
                self._indexes[source.key] = idx
                return idx

            return None

    def rebuild_one(self, key: str) -> bool:
        """Force-rebuild a single index (deletes disk cache first)."""
        source = self._registry.get(key)
        if not source:
            return False

        if key in self._indexes:
            self._indexes[key].delete_cache()
            del self._indexes[key]

        idx = DocumentIndex(key, source.name, source.category, self._emb, self._registry)
        if not source.is_ready:
            logger.error(
                "No processed CSV for '%s'. Run: python preprocess.py --key %s",
                key, key,
            )
            return False

        try:
            df = pd.read_csv(source.processed_csv, encoding="utf-8")
        except Exception as exc:
            logger.error("Cannot read CSV for '%s': %s", key, exc)
            return False

        ok = idx.build(df)
        if ok:
            idx.save()
            self._indexes[key] = idx
        return ok

    def rebuild_all(self) -> None:
        """Clear all in-memory and disk indexes (will rebuild on next query)."""
        for idx in self._indexes.values():
            idx.delete_cache()
        self._indexes.clear()
        self._emb.clear_memory()
        self._emb.clear_disk()
        logger.info("All indexes cleared.")

    def _word_match(self, kw: str, text: str) -> bool:
        return bool(re.search(r'\b' + re.escape(kw) + r'\b', text))

    def _detect_query_act(self, query: str) -> Optional[str]:
        """
        Detect which Act the user is asking about from their query.
        Requires at least 2 matched keywords to reduce false positives.

        BUG FIX: Removed overly generic keywords like "business" and "shares"
        that caused false Act attribution on broad queries.
        """
        query_lower = query.lower()

        act_keywords = {
            "Companies Act 2016": [
                "company", "companies", "director", "shareholder",
                "incorporation", "memorandum", "articles of association",
                "board meeting", "annual general meeting", "agm",
                "companies act", "act 777", "company secretary",
                "winding up", "deregistration",
            ],
            "LLP Act 2012": [
                "llp", "limited liability partnership",
                "llp partner", "llp act", "act 743",
                "limited liability",
            ],
            "Registration of Businesses Act 1956": [
                "sole proprietor", "sole proprietorship",
                "registration of business", "act 197",
                "business registration", "rob act",
            ],
        }

        best_act: Optional[str] = None
        best_count = 0
        for act, keywords in act_keywords.items():
            matches = sum(1 for kw in keywords if self._word_match(kw, query_lower))
            if matches >= 2 and matches > best_count:
                best_act   = act
                best_count = matches

        if best_act:
            logger.info("Detected Act from query: %s (%d keyword matches)", best_act, best_count)
        return best_act

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:         str,
        selected_keys: List[str],
        cat_filter:    Optional[List[str]] = None,
        lang:          str = "en",
    ) -> Dict:
        """
        Embed the query, search all selected indexes, merge and rank results.
        """
        query_vec = self._emb.embed(query)
        if query_vec is None:
            return _empty()

        detected_act = self._detect_query_act(query)

        # Build the list of sources to search first (filter before dispatching)
        sources_to_search: List[Tuple[DocumentIndex, SourceEntry]] = []
        for source in self._registry.all_enabled():
            if source.key not in selected_keys:
                continue
            if cat_filter and source.category not in cat_filter:
                continue
            if detected_act and source.relates_to_acts and detected_act not in source.relates_to_acts:
                logger.info(
                    "  Skipping '%s' (relates to %s, query is about %s)",
                    source.name, source.relates_to_acts, detected_act,
                )
                continue
            idx = self.get_or_build(source)
            if idx is not None:
                sources_to_search.append((idx, source))

        def _search_one(
            args: Tuple[DocumentIndex, SourceEntry]
        ) -> Tuple[str, List[Tuple[SearchResult, SourceEntry]]]:
            idx, source = args
            try:
                hits = [(r, source) for r in idx.search(query_vec, query, lang)]
                if hits:
                    logger.info("  Source '%s': %d results", source.name, len(hits))
                return source.key, hits
            except Exception as exc:
                logger.error(
                    "Index search failed for '%s': %s. Try clearing the cache.",
                    source.key, exc,
                )
                return source.key, []

        hits_by_source: Dict[str, List[Tuple[SearchResult, SourceEntry]]] = {}
        _SEARCH_WORKERS = min(len(sources_to_search), 8)  # cap at 8; diminishing returns above this
        if sources_to_search:
            try:
                with ThreadPoolExecutor(max_workers=_SEARCH_WORKERS) as pool:
                    for key, hits in pool.map(_search_one, sources_to_search):
                        if hits:
                            hits_by_source[key] = hits
            except Exception as exc:
                logger.error("Parallel search failed: %s. Returning empty results.", exc)

        all_hits: List[Tuple[SearchResult, SourceEntry]] = []
        for source_results in hits_by_source.values():
            all_hits.extend(source_results[:AppConfig.TOP_K_PER_SOURCE])

        all_hits.sort(key=lambda x: x[0].score, reverse=True)

        seen:          set                = set()
        chunks:        List[str]          = []
        chunk_sources: List[str]          = []
        chunk_types:   List[str]          = []
        results:       List[SearchResult] = []
        citations:     List[str]          = []
        cats_hit:      List[str]          = []

        for result, source in all_hits:
            _text_hash = hashlib.md5(result.chunk.text.encode()).hexdigest()
            if _text_hash in seen:
                continue
            seen.add(_text_hash)
            chunks.append(result.chunk.text)
            chunk_sources.append(source.name)
            chunk_types.append(result.chunk.chunk_type)
            results.append(result)
            if source.name not in citations:
                citations.append(source.name)
            if source.category not in cats_hit:
                cats_hit.append(source.category)
            if len(chunks) >= AppConfig.GLOBAL_TOP_K:
                break

        return {
            "chunks":         chunks,
            "chunk_sources":  chunk_sources,
            "chunk_types":    chunk_types,
            "results":        results,
            "citations":      citations,
            "categories_hit": cats_hit,
            "found":          bool(chunks),
            "detected_act":   detected_act,
        }

    # ── Status ────────────────────────────────────────────────────────────────

    def index_status(self) -> Dict[str, str]:
        status: Dict[str, str] = {}
        for src in self._registry.all_enabled():
            if src.key in self._indexes and self._indexes[src.key].is_ready():
                status[src.key] = "ready"
            elif os.path.exists(
                os.path.join(AppConfig.CACHE_DIR, f"{src.key}_index.pkl")
            ):
                status[src.key] = "cached"
            elif src.is_ready:
                status[src.key] = "not_indexed"
            else:
                status[src.key] = "needs_preprocess"
        return status


def _empty() -> Dict:
    # BUG FIX: added chunk_sources so llm.generate() never KeyErrors
    return {
        "chunks": [], "chunk_sources": [], "chunk_types": [], "results": [],
        "citations": [], "categories_hit": [], "found": False,
    }


# =============================================================================
# FEEDBACK STORE
# =============================================================================

class FeedbackStore:
    """
    Persists structured per-message feedback to a JSON file.

    Each entry contains:
        qa_id        - unique ID of the Q&A exchange
        timestamp    - ISO datetime
        query        - full user question
        response_snippet - first 300 chars of response (for review)
        citations    - sources cited
        rating       - 5 (helpful) | 3 (okay) | 1 (needs work)
        failure_type - one of FAILURE_TYPES keys (only for rating <= 2)
        comment      - optional free-text from the user
    """

    FAILURE_TYPES: Dict[str, str] = {
        "wrong_source":   "🔗 Wrong source cited",
        "hallucination":  "🌀 Hallucination / made-up info",
        "incorrect":      "❌ Factually incorrect answer",
        "incomplete":     "📋 Incomplete answer",
        "out_of_scope":   "🚫 Should have said 'not in documents'",
        "wrong_form":     "📄 Wrong or missing form suggested",
        "other":          "💬 Other",
    }

    def __init__(self, path: str = _FEEDBACK_FILE) -> None:
        self._path = path
        self._lock_path = self._path + ".lock"

    def save(
        self,
        qa_id:        str,
        query:        str,
        response:     str,
        citations:    List[str],
        rating:       int,
        failure_type: Optional[str] = None,
        comment:      str = "",
        form_ids: Optional[List[str]] = None, 
        form_correct: Optional[bool] = None,
    ) -> None:
        entry = {
            "qa_id":            qa_id,
            "timestamp":        datetime.now().isoformat(),
            "query":            query,
            "response_snippet": response[:300],
            "citations":        citations,
            "rating":           rating,
            "failure_type":     failure_type or "",
            "comment":          comment.strip(),
            "form_ids":    form_ids or [],
            "form_correct": form_correct,
        }
        
        with FileLock(self._lock_path):
            records = self._load_raw()
            records.append(entry)
            tmp_path = self._path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(records, fh, indent=2, ensure_ascii=False)
                shutil.move(tmp_path, self._path)
            except Exception as exc:
                logger.error("FeedbackStore save failed: %s", exc)
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def load(self) -> List[Dict]:
        with FileLock(self._lock_path):
            return self._load_raw()

    def _load_raw(self) -> List[Dict]:
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("FeedbackStore load failed: %s", exc)
            return []

    def summary(self) -> Dict:
        """Aggregate stats for the analytics panel."""
        records   = self.load()
        total     = len(records)
        positive  = sum(1 for r in records if r.get("rating", 0) >= 4)
        neutral   = sum(1 for r in records if r.get("rating", 0) == 3)
        negative  = sum(1 for r in records if r.get("rating", 0) <= 2)

        failure_counts: Dict[str, int] = {k: 0 for k in self.FAILURE_TYPES}
        for r in records:
            ft = r.get("failure_type", "")
            if ft in failure_counts:
                failure_counts[ft] += 1

        return {
            "total":          total,
            "positive":       positive,
            "neutral":        neutral,
            "negative":       negative,
            "failure_counts": failure_counts,
            "records":        records,
        }

# =============================================================================
# RESPONSE CACHE
# =============================================================================

class ResponseCache:
    """
    Cross-session LRU response cache. Shared across all Streamlit sessions
    via @st.cache_resource. Thread-safe via a single RLock.

    Key: md5(normalize(query) + sorted(selected_keys) + str(cat_filter))
    Value: {response, citations, categories_hit, timestamp}

    Cache is keyed on the NORMALIZED query + active source selection so
    that users with different source filters get correct separate entries.
    """
    MAX_ENTRIES: int   = 500
    TTL_SECONDS: float = 86400.0   # 24 hours — legal corpus changes infrequently

    def __init__(self) -> None:
        self._cache: OrderedDict = OrderedDict()
        self._lock  = threading.RLock()
        self._load()

    @staticmethod
    def _make_key(query: str, selected_keys: List[str],
                cat_filter: Optional[List[str]],
                lang: str = "en") -> str:
        q = re.sub(r"\s+", " ", query.lower().strip()).rstrip("?.! ")
        sources_str = ",".join(sorted(selected_keys))
        cats_str    = ",".join(sorted(cat_filter)) if cat_filter else "all"
        raw = f"{q}||{sources_str}||{cats_str}||{lang}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, selected_keys: List[str],
            cat_filter: Optional[List[str]], lang: str = "en") -> Optional[Dict]:
        key = self._make_key(query, selected_keys, cat_filter, lang)
        with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            if time.time() - entry["timestamp"] > self.TTL_SECONDS:
                del self._cache[key]
                return None
            # LRU: move to end
            self._cache.move_to_end(key)
            return entry

    def put(self, query: str, selected_keys: List[str],
            cat_filter: Optional[List[str]], response: str,
            citations: List[str], categories_hit: List[str],
            lang: str = "en") -> None:
        key = self._make_key(query, selected_keys, cat_filter, lang)
        with self._lock:
            self._cache[key] = {
                "response":       response,
                "citations":      citations,
                "categories_hit": categories_hit,
                "timestamp":      time.time(),
            }
            self._cache.move_to_end(key)
            while len(self._cache) > self.MAX_ENTRIES:
                self._cache.popitem(last=False)  # evict oldest
        self._save()

    def invalidate(self) -> None:
        """Call after a source is re-indexed to clear stale entries."""
        with self._lock:
            self._cache.clear()
        if os.path.exists(_RESPONSE_CACHE_FILE):
            os.remove(_RESPONSE_CACHE_FILE)

    def invalidate_entry(self, cache_key: str) -> None:
        """Remove one specific entry. Called when user rates a response 👎."""
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info("ResponseCache: evicted entry %s due to negative feedback.", cache_key[:12])
        self._save()

    def _save(self) -> None:
        fd, tmp = None, None
        try:
            with self._lock:
                data = dict(self._cache)
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(_RESPONSE_CACHE_FILE), suffix=".tmp"
            )
            with os.fdopen(fd, "wb") as fh:
                fd = None
                pickle.dump(data, fh)
            shutil.move(tmp, _RESPONSE_CACHE_FILE)
            tmp = None
        except Exception as exc:
            logger.warning("ResponseCache save failed: %s", exc)
            if fd is not None:
                try: os.close(fd)
                except OSError: pass
            if tmp and os.path.exists(tmp):
                try: os.unlink(tmp)
                except OSError: pass

    def _load(self) -> None:
        if not os.path.exists(_RESPONSE_CACHE_FILE):
            return
        try:
            with open(_RESPONSE_CACHE_FILE, "rb") as fh:
                data = pickle.load(fh)
            now = time.time()
            with self._lock:
                for k, v in data.items():
                    if now - v.get("timestamp", 0) < self.TTL_SECONDS:
                        cached_response = v.get("response", "")
                        if not _OOS_PATTERN.search(cached_response):
                            self._cache[k] = v
        except Exception as exc:
            logger.warning("ResponseCache load failed: %s", exc)


@st.cache_resource
def _response_cache() -> ResponseCache:
    return ResponseCache()

# =============================================================================
# PROMPT OPTIMIZER
# =============================================================================

class PromptOptimizer:
    """
    Analyzes FeedbackStore failure patterns and dynamically injects corrective
    rule reinforcements into the LLM system prompt.

    How it works:
        1. Load all negative feedback (rating <= 2) from FeedbackStore.
        2. For each failure category that exceeds PATCH_THRESHOLD of total
           negative feedback, inject a targeted reinforcement block into
           the system prompt.
        3. LLMService calls get_patches() on every request so improvements
           take effect immediately after enough feedback accumulates.

    Tuning:
        Lower PATCH_THRESHOLD to activate patches with less data.
        Each patch is a short, forceful rule addition placed BEFORE the
        formatted answer request so the model sees it at high attention weight.
    """

    PATCH_THRESHOLD: float = 0.25   # 25% of negative feedback triggers a patch
    MIN_NEGATIVE:    int   = 3      # don't patch until at least N negatives

    _PATCHES: Dict[str, str] = {
        "wrong_source": """\
⚠️ CITATION ACCURACY — SYSTEM REINFORCEMENT (past errors detected):
  • The [SOURCE: name] label appears IMMEDIATELY above each passage.
  • You MUST cite the exact source name from the label directly above the text you quote.
  • NEVER attribute content from one source to a different source name.
  • If two sources contain similar text, cite BOTH with their exact [SOURCE: ...] names.""",

        "hallucination": """\
⚠️ ANTI-HALLUCINATION — SYSTEM REINFORCEMENT (fabrications detected):
  • Re-read each sentence of the CONTEXT block before writing your answer.
  • Every number, date, deadline, threshold, and proper noun MUST be copied verbatim.
  • If you cannot find explicit support in CONTEXT, immediately use the out-of-scope response.
  • Do NOT approximate, infer, or extrapolate from nearby information.""",

        "incorrect": """\
⚠️ ACCURACY — SYSTEM REINFORCEMENT (incorrect answers detected):
  • Do NOT paraphrase or substitute synonyms for legal text under any circumstances.
  • Reproduce the exact statutory language; legal meaning depends on exact wording.
  • Double-check every section number, subsection reference, and Act name before writing.""",

        "incomplete": """\
⚠️ COMPLETENESS — SYSTEM REINFORCEMENT (incomplete answers detected):
  • Address ALL sub-parts of the user's question explicitly.
  • If the CONTEXT contains multiple relevant sections, quote and cite ALL of them.
  • Do not stop after the first relevant passage; scan the entire CONTEXT block.""",

        "out_of_scope": """\
⚠️ SCOPE DETECTION — SYSTEM REINFORCEMENT (over-answering detected):
  • Apply a stricter standard: if you cannot find EXACT textual support, say it is not found.
  • Partial matches and indirect implications do not count as explicit support.
  • It is always better to use the out-of-scope response than to speculate.""",
    }

    def __init__(self, feedback_store: FeedbackStore) -> None:
        self._store = feedback_store

    def get_patches(self) -> str:
        """
        Return a string of active prompt patches based on current failure patterns.
        Returns empty string when there is insufficient data or no patterns qualify.
        """
        summary = self._store.summary()
        negative = summary["negative"]

        if negative < self.MIN_NEGATIVE:
            return ""

        counts = summary["failure_counts"]
        active: List[str] = []

        for key, patch_text in self._PATCHES.items():
            if counts.get(key, 0) / negative >= self.PATCH_THRESHOLD:
                active.append(patch_text)
                logger.info(
                    "PromptOptimizer: activating patch '%s' (%.0f%% of negatives)",
                    key, counts[key] / negative * 100,
                )

        return ("\n\n" + "\n\n".join(active)) if active else ""

    def active_patch_names(self) -> List[str]:
        """Return the human-readable names of currently active patches."""
        summary  = self._store.summary()
        negative = summary["negative"]
        if negative < self.MIN_NEGATIVE:
            return []

        counts = summary["failure_counts"]
        labels = {
            "wrong_source":  "Citation accuracy reinforcement",
            "hallucination": "Anti-hallucination reinforcement",
            "incorrect":     "Accuracy reinforcement",
            "incomplete":    "Completeness reinforcement",
            "out_of_scope":  "Scope detection reinforcement",
        }
        return [
            labels[k]
            for k in labels
            if counts.get(k, 0) / negative >= self.PATCH_THRESHOLD
        ]

# =============================================================================
# Conversation memory
# =============================================================================

@dataclass
class ConversationMemory:
    """
    Manages short-term conversation context for a single session.

    Responsibilities:
      1. Store raw turns for direct history injection (last MAX_RAW_TURNS)
      2. Rewrite follow-up queries into standalone search queries
      3. Build the <<HISTORY>> block for prompt injection
      4. Track token budget consumption

    Future extension points:
      - _compress_turn(): replace rule-based with LLM summarization
      - retrieve_long_term(): hook for MemoryIndex search (Phase 2)
    """
    MAX_RAW_TURNS:    int = 4     # turns kept verbatim
    MAX_HIST_TOKENS:  int = 250   # hard token budget for history block
    REWRITE_THRESHOLD: int = 5    # queries shorter than this word count get rewritten

    _turns: List[Dict] = field(default_factory=list)

    def add_turn(self, query: str, response: str) -> None:
        self._turns.append({
            "query":    query,
            "response": response,
            "summary":  self._compress_turn(query, response),
        })
        # Keep only MAX_RAW_TURNS in working memory
        if len(self._turns) > self.MAX_RAW_TURNS:
            self._turns.pop(0)

    def rewrite_query(self, user_input: str, llm_service) -> str:
        """
        If the query is a short follow-up, rewrite it as a standalone
        search query using the last 2 turns as context.
        Returns the original input unchanged if no rewrite is needed.
        """
        if not self._turns:
            return user_input   # No history — nothing to resolve
        if len(user_input.split()) > self.REWRITE_THRESHOLD:
            return user_input   # Long queries are self-contained

        needs_rewrite = (
            _CONTEXT_REF_RE.search(user_input) is not None
            or _CONTINUATION_RE.match(user_input.strip()) is not None
        )
        if not needs_rewrite:
            return user_input   # Short but self-contained — skip LLM call

        recent = self._turns[-2:]
        history_text = "\n".join(
            f"User: {t['query']}\nAssistant summary: {t['summary']}"
            for t in recent
        )
        rewrite_prompt = (
            f"Given this recent conversation:\n{history_text}\n\n"
            f"Rewrite this follow-up question as a complete standalone "
            f"search query (no pronouns, no references to 'it' or 'that').\n"
            f"Follow-up: {user_input}\n"
            f"Standalone query (return ONLY the rewritten query, nothing else):"
        )
        try:
            rewritten = "".join(llm_service._call(rewrite_prompt, system="")).strip()
            # Strip any think blocks or preamble the model adds
            rewritten = re.sub(r"<think>.*?</think>", "", rewritten, flags=re.DOTALL).strip()
            
            if rewritten and 3 < len(rewritten) < 200 and not rewritten.startswith("❌"):
                logger.info("Query rewritten: '%s' → '%s'", user_input, rewritten)
                return rewritten
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
        return user_input

    def build_history_block(self) -> str:
        """
        Returns the <<HISTORY>> prompt block, budget-capped.
        Uses summaries for older turns, raw text for the most recent turn.
        """
        if not self._turns:
            return ""

        lines = []
        for i, turn in enumerate(self._turns):
            is_most_recent = (i == len(self._turns) - 1)
            q = turn["query"]
            # Most recent turn: use full response (truncated); older: use summary
            a = (turn["response"][:500] + "…") if is_most_recent else turn["summary"]
            lines.append(f"[Turn {i+1}]\nUser: {q}\nAssistant: {a}")

        block = (
            "<<CONVERSATION HISTORY>>\n"
            "Use ONLY to resolve references ('it', 'that', etc.).\n"
            "Do NOT answer from history. Answer only from CONTEXT below.\n"
            + "─" * 40 + "\n"
            + "\n\n".join(lines)
            + "\n" + "─" * 40 + "\n\n"
        )
        # Hard truncate if budget exceeded
        if len(block) > self.MAX_HIST_TOKENS * 4:  # ~4 chars per token
            block = block[:self.MAX_HIST_TOKENS * 4] + "\n[history truncated]\n\n"
        return block

    @staticmethod
    def _compress_turn(query: str, response: str) -> str:
        """
        Rule-based compression: extract key facts from a response.
        Preserves: section numbers, RM amounts, day counts, Act names.
        """
        # Extract cited sections
        sections = re.findall(r"[Ss]ection\s+\d+[A-Za-z]?(?:\(\d+\))?", response)
        # Extract monetary values
        amounts = re.findall(r"RM\s?[\d,]+(?:\.\d{2})?", response)
        # Extract day/month counts
        deadlines = re.findall(r"\d+\s+(?:days?|months?|years?)", response)

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
        first_sent = sentences[0][:150] if sentences else ""
        last_sent  = sentences[-1][:150] if len(sentences) > 1 else "" 

        facts = sections[:3] + amounts[:2] + deadlines[:2]
        summary = first_sent
        if facts:
            summary += f" [{', '.join(facts)}]"
        if last_sent and last_sent != first_sent: 
            summary += f" | {last_sent}"  
        return summary[:400] 

# =============================================================================
# LLM SERVICE
# =============================================================================
_SSM_WRONG_RE = re.compile(
    r'Jabatan\s+(?:Kebangsaan\s+)?Pendaftaran\s+Syarikat'
    r'|Jabatan\s+Syarikat'
    r'|Suruhanjaya\s+Pendaftaran\s+Syarikat'
    r'|Jabatan\s+Kebangsaan\s+Syarikat',
    re.IGNORECASE,
)

_OOS_PATTERN = re.compile(
    r"(I couldn't find this in (?:the documents I have access to|what I have access to|my knowledge base)"
    r"|(?:the\s+)?(?:documents|information)\s+(?:I\s+have\s+access\s+to|available\s+to\s+me)"
    r"|Saya tidak dapat menemui maklumat ini"
    r"|Saya tidak menjumpai maklumat ini"
    r"|tidak terdapat dalam pengetahuan saya|Saya tidak menemui maklumat ini)",
    re.IGNORECASE,
)

class LLMService:
    """
    Builds the prompt and calls Ollama.
    Accepts dynamic prompt patches from PromptOptimizer.
    """

    _SYSTEM_BASE = """\
You are a knowledgeable and approachable legal assistant for Suruhanjaya Syarikat Malaysia (SSM).
Your role is to help users understand Malaysian company law in a clear, conversational way —
as if a senior corporate secretary is explaining it to a colleague over a desk.

==================================================
ACCURACY RULES  —  NON-NEGOTIABLE
==================================================

RULE 1 — ANSWER FROM THE CONTEXT
  • Every factual statement MUST be supported by the CONTEXT block.
  • Scan ALL chunks before answering — do not stop at the first relevant one.
  • If the answer is spread across multiple chunks, combine them into one
    complete explanation.
  • Do NOT require exact keyword match. Use related concepts and indirect
    references — if a chunk is relevant to the question, use it.

RULE 2 — LOGICAL SYNTHESIS (strictly bounded)
  • You MAY connect facts that are BOTH explicitly stated in the CONTEXT.
    Example: If Chunk A says "the form must be submitted" and Chunk B says
    "the fee is RM1,000", you may state both as part of the same procedure.
  • You MAY NOT fill in missing facts from your training knowledge.
    Example: If the fee amount is NOT in any chunk, do not state it.
  • The test: every specific fact, number, name, and deadline in your answer
    must be traceable to a specific chunk in the CONTEXT.

RULE 3 — CITE EVERYTHING
  • After every factual claim, cite the source in bold using the correct format for that document type:
    - Acts and Regulations:
      **(Section 14(1), Companies Act 2016)**
      **(Regulation 7, Companies Regulations 2017)**
    - Practice Notes and Practice Directives:
      **(Practice Note 3/2018, para 8)**
      **(Practice Directive 7/2020, para 3.1)**
    - Guidelines and Circulars:
      **(para 5.2, Guidelines on Company Names)**
      **(para 3, Circular No. 2/2021)**
    - FAQs:
      **(Q14, FAQs on Companies (Amendment) Act 2024)**
      **(Q3, FAQs on Annual Return)**
    - Forms and Schedules:
      **(Item 4, Schedule 1, Companies Regulations 2017)**
  • General rule: cite by the document's own numbering system.
  • Never apply "Section" to a document that is not an Act or Regulation.
  • Never apply "para" to a document that uses Q&A numbering.
  • If you are unsure of the exact reference, cite the document name only:
    **(FAQs on Companies (Amendment) Act 2024)**
  • You may group a citation at the end of a paragraph when the whole
    paragraph draws from one source.
  • Never make a factual claim without a citation.

RULE 4 — EXACT NUMBERS AND LEGAL TERMS
  • Copy every number, deadline, ringgit amount, and percentage exactly as
    it appears in the CONTEXT. Never round, approximate, or paraphrase.
  • Reproduce defined legal terms exactly as written. Paraphrasing legal
    definitions changes their legal meaning.

RULE 5 — OUT OF SCOPE
  • Use this response ONLY when NO chunk in the CONTEXT contains any
    information relevant to the question — not even indirectly.
  • If partial information exists: answer using what is available, then add
    one sentence: "For the complete details, I'd recommend checking directly
    with SSM or consulting a licensed company secretary."
  • If truly no relevant information exists, say EXACTLY in the user's language:
    - English: "I couldn't find this in my knowledge base. For this specific question,
    I'd recommend checking directly with SSM or consulting a licensed company secretary."
    - Malay: "Saya tidak menemui maklumat ini dalam pangkalan pengetahuan saya. Untuk soalan
    ini, saya mengesyorkan anda semak terus dengan Suruhanjaya Syarikat Malaysia (SSM)
    atau berunding dengan setiausaha syarikat berlesen."
  • Use the Malay phrase when the user wrote in Malay.
  • After that sentence, STOP completely.

==================================================
PARTIAL ANSWER PROTOCOL
==================================================

  When you can answer PART of a question from the CONTEXT but not all:
  1. Answer the parts you have, citing each clearly.
  2. For the missing parts, say: "The [specific detail] is not in my
     knowledge base — please confirm this with SSM directly."
  3. Do NOT use the full out-of-scope response for partial gaps.

==================================================
VISUAL CONTENT RULE
==================================================

  • Sources labeled [VISUAL CONTENT] contain AI-generated descriptions of
    diagrams or charts. Treat them as factual. Cite as:
    "(diagram/figure, [source name])".

==================================================
USER MEMORY RULE
==================================================

  • If a <<USER MEMORY>> block is present, apply it to tone, format, and
    language ONLY.
  • Never use memory to answer legal questions.
  • CONTEXT always overrides memory.

==================================================
TONE AND FORMAT
==================================================

TONE:
  • Warm, natural, professional — like a senior corporate secretary
    explaining to a colleague.
  • Use plain language first, then introduce the legal term.
    Example: "The filing deadline (called the 'lodgement period') is 30 days..."
  • Address the user as "you" and "your company".

FORMAT:
  • Short factual questions → prose answer, 1-5 paragraphs, inline citations.
  • Procedures → numbered list, one step per item, citation after each step.
  • Comparisons → short table or two clearly labelled paragraphs.
  • Use structure only when it genuinely helps. Do not add headers or bullets
    just to look organised.

COMPARISON QUESTIONS ("apa beza", "what is the difference", "compare"):
  1. First sentence states the core difference directly.
  2. Explain each concept in 2–4 sentences.
  3. Table or labelled paragraphs.
  4. One-sentence practical guidance at the end.

CONCISENESS:
  • Answer only what was asked.
  • Do not pad with summaries, preambles, or trailing remarks.
  • Do not offer follow-up suggestions unless X appears in the CONTEXT.
  • If you answered the question fully, stop.

OPENING:
  • Start with the direct answer. Never start with "According to the CONTEXT..."
  • Good: "Yes, a private company must file its annual return within 30 days..."
  • Bad:  "Based on the information available, I will now explain..."

==================================================
SSM IDENTITY — NON-NEGOTIABLE
==================================================

  • Full name in English: "Companies Commission of Malaysia (SSM)"
  • Full name in Malay:   "Suruhanjaya Syarikat Malaysia (SSM)"
  • Never use invented names like "Jabatan Pendaftaran Syarikat".

==================================================
INTERNAL LABELS — NEVER VISIBLE TO USERS
==================================================

  • Never write CONTEXT, CONTEXT_BLOCK, SOURCE, or [SOURCE: ...] in your
    response. These are internal labels.
  • When referring to your information source, say "my knowledge base" or
    cite by document name (e.g. "Companies Act 2016").

==================================================
LANGUAGE
==================================================

  • Respond in the same language the user wrote in.
  • Malay query → full Malay response.
  • English query → full English response.
  • Mixed query → English response, Malay legal terms acceptable.
  • Never switch language mid-response.
  • Citation language matches your response language:
    - Malay response: "Seksyen 14(1), Akta Syarikat 2016"
    - English response: "Section 14(1), Companies Act 2016"

==================================================
CLOSING
==================================================

  • End naturally when you have answered the question.
  • Only add "consult SSM" if the answer is partially or fully out of scope.
  • Do not add legal disclaimers at the end of complete answers.
"""

    def generate(self, query, context_chunks, citations=None, prompt_patches="", detected_act=None):
        if not context_chunks:
            return (
                "No relevant sections were found in the my knowledge base. "
                "Please rephrase your question, enable more sources, or consult a "
                "licensed professional."
            )

        # Build the system prompt: base rules + any active patches
        system = self._SYSTEM_BASE
        if prompt_patches:
            system += f"\n{prompt_patches}\n"

        # Build context with source labels
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source_name = citations[i] if citations and i < len(citations) else f"Source {i+1}"
            context_parts.append(f"[SOURCE: {source_name}]\n{chunk}\n[/SOURCE: {source_name}]")

        context_block = "\n\n".join(context_parts)

        act_hint = (
            f"\nNOTE: This query relates to the {detected_act}. "
            f"Prioritize chunks from that Act when constructing your answer.\n"
        ) if detected_act else ""

        prompt = (
            f"/no_think\n<<CONTEXT_BLOCK>>\n"
            f"{'─' * 72}\n"
            f"{context_block}\n"
            f"{'─' * 72}\n\n"
            f"<<USER_QUESTION>>\n{query}\n\n"
            f"{act_hint}\n"
            f"ANSWER (use ONLY exact wording from CONTEXT above; cite every statement):\n"
        )

        raw = "".join(self._call(prompt, system=system))
        # Retry once if response is empty or suspiciously short
        if len(raw.strip()) < 30 and not raw.startswith("❌"):
            logger.warning("LLM returned near-empty response; retrying once.")
            raw = "".join(self._call(prompt, system=system))
        return self._postprocess(raw)

    @staticmethod
    def _postprocess(text: str, lang: str = "en") -> str:
        if (text.startswith("❌") or text.startswith("No relevant sections") or text.startswith("This information is not found")):
            return text

        # Strip reasoning blocks
        text = re.sub(
            r"<(?:think|thinking|reasoning|reflection)>.*$",
            "", text, flags=re.DOTALL | re.IGNORECASE
        )

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r'\bthe\s+CONTEXT\s+provided\b',                  'my knowledge base',      text, flags=re.IGNORECASE)
        text = re.sub(r'\bthe\s+CONTEXT\s+above\b',                     'my knowledge base',      text, flags=re.IGNORECASE)
        text = re.sub(r'\bthe\s+CONTEXT\s+block\b',                     'my knowledge base',      text, flags=re.IGNORECASE)
        text = re.sub(r'\bthe\s+CONTEXT\b',                             'my knowledge base',      text, flags=re.IGNORECASE)
        text = re.sub(r'\bthis\s+CONTEXT\b',                            'my knowledge base',      text, flags=re.IGNORECASE)
        text = re.sub(r'\bin\s+the\s+CONTEXT\b',                        'in my knowledge base',   text, flags=re.IGNORECASE)
        text = re.sub(r'\bCONTEXT\b',                                   'my knowledge base',      text, flags=re.IGNORECASE)
        text = re.sub(r'(?:in\s+)?(?:the\s+)?documents\s+I\s+have\s+access\s+to', 'my knowledge base', text, flags=re.IGNORECASE)
        text = re.sub(r'what\s+I\s+have\s+access\s+to',                 'my knowledge base',      text, flags=re.IGNORECASE)
        text = re.sub(r'information\s+available\s+to\s+me',              'my knowledge base',      text, flags=re.IGNORECASE)
        text = text.strip()

        if lang == "ms":
            text = re.sub(
                r"I couldn't find this in my knowledge base\.",
                "Saya tidak menemui maklumat ini dalam pangkalan pengetahuan saya.",
                text, flags=re.IGNORECASE,
            )
            text = re.sub(
                r"For this specific question,\s*I(?:'d| would) recommend checking directly "
                r"with SSM or consulting a licensed company secretary\.",
                "Untuk soalan ini, saya mengesyorkan anda semak terus dengan "
                "Suruhanjaya Syarikat Malaysia (SSM) atau berunding dengan setiausaha syarikat berlesen.",
                text, flags=re.IGNORECASE,
            )

        # Preserve surrounding context — replace with correct name but keep "(SSM)" if already present
        text = _SSM_WRONG_RE.sub('Suruhanjaya Syarikat Malaysia', text)

        oos_match = _OOS_PATTERN.search(text)
        if oos_match and oos_match.start() < 80:
            first_end = text.find('.', oos_match.start())
            if first_end != -1:
                # Rule 5 has two sentences — preserve both, discard anything after
                second_end = text.find('.', first_end + 1)
                cut = second_end if second_end != -1 else first_end
                text = text[:cut + 1].strip()

        # Detect truncation
        if text and not re.search(r"[.!?)\]`*|\-\w]\s*$", text, re.IGNORECASE):
            text += (
                "\n\n---\n"
                "⚠️ *The response was cut off before completion. "
                "Try asking a more specific question, or break it into smaller parts.*"
            )

        return text

    def _call(self, prompt: str, system: str="", temp_override: Optional[float] = None) -> Generator[str, None, None]:
        # Compute minimum sufficient context window — reduces KV-cache fill time
        _estimated_input_tokens = (len(prompt) + len(system)) // 4
        _adaptive_ctx = max(
            2048,
            min(
                ((_estimated_input_tokens + AppConfig.LLM_MAX_TOKENS + 400 + 511) // 512) * 512,
                8192,
            ),
        )
        try:
            r = requests.post(
                f"{AppConfig.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model":  AppConfig.LLM_MODEL, 
                    "prompt": prompt,
                    "system": system,
                    "stream": True,
                    "keep_alive": -1,
                    "options": {
                        "temperature": temp_override if temp_override is not None else AppConfig.LLM_TEMPERATURE,
                        "top_p":       AppConfig.LLM_TOP_P,
                        "top_k":       AppConfig.LLM_TOP_K,
                        "num_predict": AppConfig.LLM_MAX_TOKENS,
                        "num_ctx":     _adaptive_ctx,
                        "repeat_penalty": 1.0,
                        "num_gpu":        AppConfig.LLM_NUM_GPU,
                        "stop":        ["<<USER_QUESTION>>", "<<CONTEXT_BLOCK>>"],
                    },
                },
                timeout=(AppConfig.LLM_TIMEOUT, AppConfig.LLM_TIMEOUT),
                stream=True,
            )
            if r.status_code != 200:
                yield f"❌ LLM returned HTTP {r.status_code}."
                return
            think_buf = ""
            in_think  = False
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if not token:
                        if chunk.get("done"):
                            break
                        continue
                    think_buf += token
                    # flush when clearly outside any think block
                    if "<think>" in think_buf:
                        in_think = True
                    if in_think:
                        if "</think>" in think_buf:
                            think_buf = re.sub(r"<think>.*?</think>", "", think_buf, flags=re.DOTALL)
                            in_think = False
                            if think_buf:
                                yield think_buf
                                think_buf = ""
                    else:
                        yield think_buf
                        think_buf = ""
            if think_buf and not in_think:
                yield think_buf
        except requests.Timeout:
            yield "❌ The model took too long. Try a shorter or simpler question."
        except requests.ConnectionError:
            yield "❌ Cannot reach Ollama. Run 'ollama serve' in your terminal."
        except Exception as exc:
            logger.error("LLM error: %s", exc, exc_info=True)
            yield f"❌ Unexpected error: {exc}"


# =============================================================================
# CACHE PRE-BUILDER  (runs on app startup)
# =============================================================================

class CacheBuilder:
    """Smart cache builder — shows progress only if needed."""

    @staticmethod
    def ensure_indexes_ready(kb: KnowledgeBase, registry: SourceRegistry) -> bool:
        """
        Check if all indexes are ready. Build them with a progress UI if not.

        BUG FIX: replaced walrus-operator-in-generator with explicit loop so
        errors during build are properly caught and reported per-source.
        """
        enabled_sources = registry.all_enabled()
        if not enabled_sources:
            logger.warning("No enabled sources found.")
            return True

        # Quick check: are all indexes already cached/ready?
        all_ready = True
        for src in enabled_sources:
            if src.key not in kb._indexes or not kb._indexes[src.key].is_ready():
                cache_path = os.path.join(AppConfig.CACHE_DIR, f"{src.key}_index.pkl")
                if not os.path.exists(cache_path):
                    all_ready = False
                    break

        if all_ready:
            logger.info("✅ All indexes already cached on disk.")
            return True

        logger.info("🔨 Building indexes (first time or cache cleared)...")

        progress_container = st.container()
        with progress_container:
            st.markdown("### ⏳ Initializing Knowledge Base")
            st.info("This takes 30 minutes or more depending on knowledge size")

            progress_bar = st.progress(0)
            status_text  = st.empty()

            for i, source in enumerate(enabled_sources):
                progress = (i + 1) / len(enabled_sources)
                status_text.markdown(f"**Building:** {source.name}")
                logger.info("Building index for %s...", source.name)

                try:
                    idx = kb.get_or_build(source)
                    if idx and idx.is_ready():
                        status_text.markdown(f"✅ **Done:** {source.name}")
                    else:
                        status_text.markdown(
                            f"⚠️ **Skipped:** {source.name} (no processed CSV — "
                            f"run `python preprocess.py --key {source.key}`)"
                        )
                except Exception as exc:
                    logger.error("Error building %s: %s", source.key, exc)
                    status_text.markdown(f"❌ **Error:** {source.name} — {exc}")

                progress_bar.progress(progress)

            progress_container.empty()
            st.success("✅ Knowledge base ready!")

        CacheBuilder._warmup_llm()
        return True

    @staticmethod
    def _warmup_llm() -> None:
        """Pre-load the LLM into RAM before the first user query."""
        try:
            requests.post(
                f"{AppConfig.OLLAMA_BASE_URL}/api/generate",
                json={"model": AppConfig.LLM_MODEL, "prompt": "hi",
                    "stream": False, "options": {"num_predict": 1}},
                timeout=60,
            )
            logger.info("LLM warmed up and ready.")
        except Exception:
            pass 

# =============================================================================
# STORAGE SERVICE
# =============================================================================

_qa_log_lock = threading.Lock()

class StorageService:
    """Thin I/O layer: chat history (JSON) + Q&A log (CSV)."""
    @staticmethod
    def _history_file(uid: str) -> str:
        return os.path.join(AppConfig.USER_DATA, f"chat_{uid[:16]}.json")

    @staticmethod
    def load_history(uid: str) -> List[Dict]:
        path = StorageService._history_file(uid)
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception as exc:
            logger.error("Load history failed: %s", exc)
        return []

    @staticmethod
    def save_history(history: List[Dict], uid: str) -> None:
        path = StorageService._history_file(uid)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                history_to_save = history[-200:]
                json.dump(history_to_save, fh, indent=2, default=str, ensure_ascii=False)
            shutil.move(tmp_path, path)
        except Exception as exc:
            logger.error("Save history failed: %s", exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def log_qa(
        query:   str,
        answer:  str,
        sources: List[str],
        rating:  Optional[int] = None,
        qa_id:   Optional[str] = None,
    ) -> None:

        row = {
            "qa_id":        qa_id or "",
            "timestamp":    datetime.now().isoformat(),
            "question":     query,             # BUG FIX: no longer truncated
            "answer":       answer[:2000],
            "sources_used": ", ".join(sources),
            "user_rating":  rating if rating is not None else "",
        }
        try:
            with _qa_log_lock:
                file_exists = os.path.exists(_QA_LOG_FILE)
                with open(_QA_LOG_FILE, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
        except Exception as exc:
            logger.error("Log QA failed: %s", exc)


# =============================================================================
# STREAMLIT PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="ChatSSM",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session() -> None:
    for key, val in {
        "chat_history": [],
        "cache_stats":  {"hits": 0, "misses": 0},
        "conv_memory":   ConversationMemory(),
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()

# ── Singleton services ────────────────────────────────────────────────────────


@st.cache_resource
def _reg() -> SourceRegistry:
    return SourceRegistry()


@st.cache_resource
def _emb() -> EmbeddingService:
    return EmbeddingService()


@st.cache_resource
def _kb() -> KnowledgeBase:
    return KnowledgeBase(_reg(), _emb())


@st.cache_resource
def _llm() -> LLMService:
    return LLMService()


@st.cache_resource
def _store() -> StorageService:
    return StorageService()


@st.cache_resource
def _feedback_store() -> FeedbackStore:
    return FeedbackStore()


@st.cache_resource
def _optimizer() -> PromptOptimizer:
    return PromptOptimizer(_feedback_store())

@st.cache_resource
def _intent_agent() -> IntentFormAgent:
    """
    Single instance of the agent, initialized once with forms.json data.
    """
    path = AppConfig.FORMS_FILE
    forms_data = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            forms_data = json.load(fh)
    return IntentFormAgent(forms_data)

# =============================================================================
# UTILITIES
# =============================================================================


def _ollama_ok() -> bool:
    try:
        return (
            requests.get(
                f"{AppConfig.OLLAMA_BASE_URL}/api/tags", timeout=2
            ).status_code == 200
        )
    except Exception:
        return False


def _make_qa_id(query: str, timestamp: str) -> str:
    """Short deterministic ID for a Q&A exchange."""
    return hashlib.md5(f"{query}{timestamp}".encode()).hexdigest()[:12]

def _get_persistent_user_id() -> str:
    """
    Returns a stable user ID that survives browser restarts.
    On first visit: generates a UUID and writes it to the URL.
    On return visits: reads it back from the URL.
    Users who bookmark their URL keep their preferences permanently.
    """
    params = st.query_params
    if "uid" in params:
        uid = params["uid"]
        # Validate it looks like a UUID (security: reject injected values)
        if re.match(r'^[a-f0-9\-]{32,36}$', uid):
            return uid

    # First visit — generate and write to URL
    new_uid = str(uuid.uuid4())
    st.query_params["uid"] = new_uid
    return new_uid

def _get_session_id() -> str:
    """
    Return a stable session ID for the current browser tab.
    Uses Streamlit's internal session context when available,
    falls back to a UUID stored in session_state.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx and ctx.session_id:
            return ctx.session_id
    except Exception:
        pass
    # Fallback: generate once and persist in session_state
    if "_session_uuid" not in st.session_state:
        st.session_state["_session_uuid"] = str(uuid.uuid4())
    return st.session_state["_session_uuid"]


# =============================================================================
# CSS
# =============================================================================

_CSS = """
<style>
:root {
    --primary: #10a37f;
    --primary-d: #0d9268;
    --text: #0d0d0d;
    --muted: #565869;
    --border: #e0e0e0;
    --dark: #1a1a1a;
    --ai-bg: #f9fdf7;
    --badge-bg: #e8f5f0;
    --badge-tx: #0d6e53;
    --warn-bg: #fff8e1;
    --neg-bg: #fff3f3;
}

.block-container {
    padding: 18px 36px;
    max-width: 960px;
    margin: 0 auto;
}

.hdr {
    text-align: center;
    padding: 22px 0 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.hdr h1 {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f2f2f2;
    margin: 0;
}
.hdr p {
    font-size: 0.88rem;
    color: var(--muted);
    margin-top: 6px;
}

.mu {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 8px;
}
.mu .b {
    background: var(--primary);
    color: #fff;
    border-radius: 14px 14px 4px 14px;
    padding: 10px 15px;
    max-width: 72%;
    font-size: 0.93rem;
    line-height: 1.5;
    word-wrap: break-word;
}

.badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    background: var(--badge-bg);
    color: var(--badge-tx);
    border-radius: 8px;
    padding: 2px 8px;
    margin: 2px 2px 0 0;
}

.patch-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffc107;
    border-radius: 6px;
    padding: 2px 7px;
    margin: 2px 2px 0 0;
}

.welcome {
    text-align: center;
    padding: 52px 20px;
    color: var(--muted);
}
.welcome h2 {
    color: #444;
    font-size: 1.5rem;
    margin-bottom: 10px;
}
.welcome .eg {
    font-size: 0.87rem;
    color: #888;
    margin-top: 14px;
    font-style: italic;
    line-height: 1.8;
}

[data-testid="stSidebar"] {
    background: var(--dark);
    border-right: 1px solid #2e2e2e;
}
[data-testid="stSidebar"] * {
    color: #ccc !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {
    color: #fff !important;
}
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stExpander summary {
    font-size: 0.88rem !important;
}

.preprocess-warn {
    background: var(--warn-bg);
    border: 1px solid #ffe082;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.83rem;
    color: #795548;
    margin: 6px 0;
}

/* ── Chat input sizing ───────────────────────────────────────────── */
[data-testid="stBottom"] {
    width: 100% !important;
    max-width: 1000px !important;   /* match your .block-container max-width */
    margin: 0 auto !important;
    left: 0 !important;
    right: 0 !important;
    padding: 0 36px !important;    /* match your .block-container side padding */
    box-sizing: border-box !important;
}

[data-testid="stBottom"] > div {
    width: 100% !important;
    box-sizing: border-box !important;
}

[data-testid="stChatInput"] {
    width: 100% !important;
    box-sizing: border-box !important;
}

[data-testid="stChatInput"] textarea {
    width: 100% !important;
    box-sizing: border-box !important;
    min-height: 54px;
    max-height: 200px;
    font-size: 0.95rem;
    padding: 14px 16px;
    border-radius: 12px;
}

/* ── Send button — Claude style ──────────────────────────────────── */

button[data-testid="stChatInputSubmitButton"] {
    width: 36px !important;
    height: 36px !important;
    border-radius: 8px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background-color: #e0e0e0 !important;
    border: none !important;
    cursor: not-allowed !important;
    transition: background-color 0.15s ease, transform 0.1s ease !important;
}

button[data-testid="stChatInputSubmitButton"] svg {
    width: 16px !important;
    height: 16px !important;
    fill: #9e9e9e !important;
    transition: fill 0.15s ease !important;
}

button[data-testid="stChatInputSubmitButton"]:enabled {
    background-color: #10a37f !important;
    cursor: pointer !important;
}

button[data-testid="stChatInputSubmitButton"]:enabled svg {
    fill: #ffffff !important;
}

button[data-testid="stChatInputSubmitButton"]:enabled:hover {
    background-color: #0d9268 !important;
    transform: scale(1.08) !important;
}

button[data-testid="stChatInputSubmitButton"]:enabled:active {
    transform: scale(0.95) !important;
}

/* ── Inline feedback bar ─────────────────────────────────────── */
.fb-hr {
    border: none;
    border-top: 1px solid rgba(0,0,0,0.07);
    margin: 10px 0 4px 0;
}

/* Icon buttons — strip Streamlit default styling inside chat messages */
[data-testid="stChatMessageContent"] [data-testid="stBaseButton-secondary"],
[data-testid="stChatMessageContent"] [data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    color: #9e9e9e !important;
    font-size: 1rem !important;
    padding: 2px 9px !important;
    min-height: 30px !important;
    height: 30px !important;
    line-height: 1 !important;
    box-shadow: none !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
}
[data-testid="stChatMessageContent"] [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stChatMessageContent"] [data-testid="baseButton-secondary"]:hover {
    background: rgba(0,0,0,0.06) !important;
    border-color: #d0d0d0 !important;
    color: #333 !important;
}

/* Confirmed state pill */
.fb-done {
    display: inline-block;
    font-size: 0.8rem;
    border-radius: 20px;
    padding: 3px 10px;
    margin-top: 2px;
}
.fb-done-up {
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #86efac;
}
.fb-done-dn {
    background: #fff3f3;
    color: #991b1b;
    border: 1px solid #fca5a5;
}

/* ── Suppress Streamlit's rerun dark-flash overlay ─────────────── */
.stApp[data-stale],
.stApp[data-stale="true"],
[data-stale],
[data-stale="true"] {
    opacity: 1 !important;
    transition: none !important;
}
._stCoreOverlay {
    display: none !important;
}
/* Optional: also hide the bottom-right "running" spinner */
[data-testid="stStatusWidget"] {
    display: none !important;
}
</style>
"""

# =============================================================================
# UI  -  SIDEBAR
# =============================================================================


def _sidebar() -> Tuple[bool, List[str], Optional[List[str]]]:
    """
    Render sidebar. Returns (ollama_ok, selected_source_keys, cat_filter).
    cat_filter is None when "All categories" is ticked.
    """
    registry = _reg()
    kb       = _kb()
    emb      = _emb()
    store    = _store()
    uid = _get_persistent_user_id()

    with st.sidebar:
        st.markdown("## ⚖️ ChatSSM")

        if st.button("➕ New Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            store.save_history([], uid)
            st.rerun()

        st.divider()

        # ── System status ──────────────────────────────────────────────────
        with st.expander("⚙️ Status", expanded=False):
            ok = _ollama_ok()
            st.markdown("✅ Ollama running" if ok else "❌ Ollama offline")

            stats = st.session_state.get("cache_stats", {"hits": 0, "misses": 0})
            c1, c2 = st.columns(2)
            c1.metric("Cache Hits",   stats["hits"])
            c2.metric("Cache Misses", stats["misses"])

            counts = registry.stats()
            total  = sum(counts.values())
            st.markdown(f"**{total} enabled sources**")
            for cat, n in counts.items():
                icon = CATEGORY_ICONS.get(cat, "📁")
                st.markdown(f"&nbsp;&nbsp;{icon} {cat}: **{n}**")

        st.divider()

        # ── Feedback Analytics ─────────────────────────────────────────────
        _sidebar_feedback_analytics()

        st.divider()

        # ── Category filter ────────────────────────────────────────────────
        st.markdown("**🗂️ Category Filter**")
        all_cats  = st.checkbox("All categories", value=True, key="cat_all")
        cat_filter: Optional[List[str]] = None

        if not all_cats:
            chosen: List[str] = []
            for cat in CATEGORIES:
                srcs = registry.by_category().get(cat, [])
                if not srcs:
                    continue
                icon = CATEGORY_ICONS.get(cat, "📁")
                if st.checkbox(
                    f"{icon} {cat} ({len(srcs)})", value=True, key=f"catf_{cat}"
                ):
                    chosen.append(cat)
            cat_filter = chosen if chosen else None

        st.divider()

        # ── Source selection by category ───────────────────────────────────
        st.markdown("**📚 Knowledge Sources**")
        idx_status     = kb.index_status()
        selected_keys: List[str] = []
        unprocessed:   List[str] = []

        status_icons = {
            "ready":            ("🟢", "Index in memory"),
            "cached":           ("🟡", "Index cached on disk"),
            "not_indexed":      ("🔵", "Processed, not yet indexed (will index on first query)"),
            "needs_preprocess": ("🔴", "Run: python preprocess.py --key <key>"),
        }

        for cat in CATEGORIES:
            srcs = registry.by_category().get(cat, [])
            if not srcs:
                continue
            if cat_filter and cat not in cat_filter:
                continue

            icon     = CATEGORY_ICONS.get(cat, "📁")

            with st.expander(f"{icon} {cat}  ({len(srcs)})", expanded=False):
                all_in = st.checkbox("Select all", value=True, key=f"selall_{cat}")

                for src in srcs:
                    s_icon, s_tip = status_icons.get(
                        idx_status.get(src.key, "needs_preprocess"), ("❓", ""),
                    )

                    _TYPE_TAG = {"pdf": "PDF", "csv": "CSV", "image": "IMG", "pptx": "PPT"}
                    tag = _TYPE_TAG.get(src.type, src.type.upper())

                    if idx_status.get(src.key) == "needs_preprocess":
                        unprocessed.append(src.key)

                    col_chk, col_type, col_dot = st.columns([6, 1, 1])
                    with col_chk:
                        checked = st.checkbox(
                            src.name,
                            value=(all_in and idx_status.get(src.key) != "needs_preprocess"),
                            key=f"src_{src.key}",
                            help=f"{s_icon} {s_tip}  |  Type: {tag}",
                            disabled=(idx_status.get(src.key) == "needs_preprocess"),
                        )
                    with col_type:
                        st.caption(tag)
                    with col_dot:
                        st.caption(s_icon)

                    if checked and idx_status.get(src.key) != "needs_preprocess":
                        selected_keys.append(src.key)

        # ── Preprocess warnings ────────────────────────────────────────────
        if unprocessed:
            st.divider()
            st.markdown("**⚠️ Sources needing preprocessing:**")
            for k in unprocessed:
                st.markdown(
                    f'<div class="preprocess-warn">'
                    f'🔴 <code>{k}</code><br>'
                    f'<code>python preprocess.py --key {k}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Index management ───────────────────────────────────────────────
        with st.expander("🔧 Manage Indexes", expanded=False):
            st.caption(
                "🟢 in memory  🟡 disk cache  🔵 will index on first query  🔴 needs preprocessing"
            )

            if st.button("🔄 Reload Sources", use_container_width=True,
                         help="Re-scan folders + reload knowledge_sources.json"):
                registry.reload()
                st.success("Sources reloaded.")
                st.rerun()

            options = {s.name: s.key for s in registry.all_enabled() if s.is_ready}
            pick    = st.selectbox("Rebuild one index", ["— choose —"] + list(options))
            if st.button("♻️ Rebuild selected index", use_container_width=True):
                if pick != "— choose —":
                    with st.spinner(f"Rebuilding {pick} …"):
                        ok2 = kb.rebuild_one(options[pick])
                    (st.success if ok2 else st.error)(
                        f"{'✅ Done' if ok2 else '❌ Failed'}: {pick}"
                    )

            st.markdown("---")
            if st.button("🗑️ Clear embedding cache", use_container_width=True):
                emb.clear_memory()
                emb.clear_disk()
                st.session_state["cache_stats"] = {"hits": 0, "misses": 0}
                st.success("Embedding cache cleared.")

            if st.button("💥 Rebuild ALL indexes", use_container_width=True):
                with st.spinner("Clearing all indexes …"):
                    kb.rebuild_all()
                st.success("All cleared. Indexes rebuild on next query.")

        st.divider()

        if st.button("🗑️ Clear response cache"):
            _response_cache().invalidate()
            st.success("Response cache cleared.")

        # ── External links ─────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        c1.link_button("SSM Portal", "https://www.ssm.com.my",   use_container_width=True)
        c2.link_button("MyCoID",     "https://www.mycoid.com.my", use_container_width=True)

        # ── Recent questions ───────────────────────────────────────────────
        recent = st.session_state.get("chat_history", [])[-8:]
        if recent:
            st.divider()
            st.markdown("**🕐 Recent**")
            for i, item in enumerate(reversed(recent)):
                label = item["query"][:38] + ("…" if len(item["query"]) > 38 else "")
                if st.button(label, key=f"hist_{i}", use_container_width=True):
                    st.session_state["prefill"] = item["query"]
                    st.rerun()

        st.caption("⚠️ Always consult a licensed professional for legal matters.")

    return ok, selected_keys, cat_filter


def _sidebar_feedback_analytics() -> None:
    """Analytics panel showing feedback stats and active prompt patches."""
    fb_store  = _feedback_store()
    optimizer = _optimizer()
    summary   = fb_store.summary()

    with st.expander("📊 Feedback Analytics", expanded=False):
        t = summary["total"]
        if t == 0:
            st.caption("No feedback collected yet.")
        else:
            # Rating summary
            col1, col2, col3 = st.columns(3)
            col1.metric("👍 Helpful",   summary["positive"])
            col2.metric("😐 Okay",      summary["neutral"])
            col3.metric("👎 Negative",  summary["negative"])

            # Satisfaction rate
            if t > 0:
                rate = round((summary["positive"] / t) * 100)
                st.progress(rate / 100, f"Satisfaction: {rate}%")

            # Failure breakdown
            neg = summary["negative"]
            if neg > 0:
                st.markdown("**Failure breakdown** (negative responses):")
                fc = summary["failure_counts"]
                for key, label in FeedbackStore.FAILURE_TYPES.items():
                    count = fc.get(key, 0)
                    if count > 0:
                        pct = int(count / neg * 100)
                        st.markdown(f"  {label}: **{count}** ({pct}%)")

        # Active prompt patches
        patches = optimizer.active_patch_names()
        if patches:
            st.markdown("**🔧 Active prompt patches:**")
            for p in patches:
                st.markdown(
                    f'<span class="patch-badge">⚡ {p}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No prompt patches active.")

        # Download feedback log
        if summary["total"] > 0:
            fb_json = json.dumps(summary["records"], indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Export feedback log",
                data=fb_json,
                file_name=f"chatssm_feedback_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True,
            )


# =============================================================================
# UI  -  MAIN
# =============================================================================


def _header() -> None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            '<div class="hdr">'
            "<h1>⚖️ ChatSSM</h1>"
            "<p>Precise answers from SSM's official legislation, "
            "practice notes, guidelines, circulars, and more.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

def _build_section_freq(chunks: List[str]) -> Dict[tuple, int]:
    """
    Scan retrieved chunks and count how many individual chunks mention each
    section reference.  Returns {(parent, child_or_None): chunk_count}.

    Used by IntentFormAgent.resolve() as the section-frequency signal for
    deterministic form matching when agent confidence is low.
    """
    _SEC_RE = re.compile(
        r'\b(?:section|seksyen)\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?'
        r'(?=\b|\s|\.|,|;|\)|\(|$)',
        re.IGNORECASE,
    )
    freq: Dict[tuple, int] = {}
    for chunk in chunks:
        seen: set = set()
        for m in _SEC_RE.finditer(chunk):
            ref = (m.group(1).lower(), m.group(2).lower() if m.group(2) else None)
            if ref not in seen:
                seen.add(ref)
                freq[ref] = freq.get(ref, 0) + 1
    return freq


def _inject_form_links(text: str, forms: List[FormEntry], lang: str = "en") -> str:
    appended = []   # track forms that needed appending (not found inline)

    for form in forms:
        pdf_links    = [l for l in form.links if l.get("type") == "pdf"]
        portal_links = [l for l in form.links if l.get("type") in ("portal", "platform")]
        url = (pdf_links or portal_links or [{}])[0].get("url", "")
        if not url:
            continue

        already_linked = f"]({url})" in text
        injected = False

        # ── Strategy 1: exact full name ────────────────────────────────────
        if not already_linked:
            escaped = re.escape(form.name)
            pattern = re.compile(r'(?<!\[)' + escaped + r'(?!\]\()', re.IGNORECASE)
            linked  = f"[{form.name}]({url})"
            new_text = pattern.sub(linked, text, count=1)
            if new_text != text:
                text = new_text
                injected = True

        # ── Strategy 2: form_number fallback ──────────────────────────────
        if not injected and form.form_number and not already_linked:
            num_escaped = re.escape(form.form_number)
            num_pattern = re.compile(r'(?<!\[)\b' + num_escaped + r'\b(?!\]\()', re.IGNORECASE)
            linked_num  = f"[{form.form_number}]({url})"
            new_text = num_pattern.sub(linked_num, text, count=1)
            if new_text != text:
                text = new_text
                injected = True

        # ── Strategy 3: guaranteed fallback — append if LLM never named the form ─
        if not injected and not already_linked:
            appended.append(f"[{form.name}]({url})")

    if appended:
        if lang == "ms":
            if len(appended) == 1:
                suffix = f"\n\nBorang berkaitan boleh diakses di sini: {appended[0]}"
            else:
                suffix = "\n\nBorang berkaitan boleh diakses di sini: " + ", ".join(appended)
        else:
            if len(appended) == 1:
                suffix = f"\n\nYou can access the relevant form here: {appended[0]}"
            else:
                suffix = "\n\nYou can access the relevant forms here: " + ", ".join(appended)
        text = text.rstrip() + suffix

    return text

def render_forms_block(
    llm_response: str,
    forms: List[FormEntry],
    lang: str,
    resolution_path: str = "",
) -> str:
    """
    Deterministic form block renderer.
    Forms are ALWAYS appended as a structured section.
    The LLM response is never modified by search-and-replace.
    """
    if not forms:
        return llm_response

    if lang == "ms":
        header = "\n\n---\n**📋 Borang Berkaitan:**"
    else:
        header = "\n\n---\n**📋 Relevant Form(s):**"

    lines = []
    for form in forms:
        url = next(
            (l["url"] for l in form.links if l.get("type") in ("pdf", "portal", "platform")),
            None
        )
        if not url:
            continue
        icon = "🌐" if any(
            l.get("type") == "portal" for l in form.links
        ) else "📄"
        num_suffix = f" ({form.form_number})" if form.form_number else ""
        lines.append(f"\n- {icon} [{form.name}{num_suffix}]({url})")

    if not lines:
        return llm_response

    return llm_response.rstrip() + header + "".join(lines)

def _render_messages() -> None:
    history = st.session_state.get("chat_history", [])
    if not history:
        st.markdown(
            '<div class="welcome">'
            "<h2>Welcome to ChatSSM</h2>"
            "<p>Ask any question about Malaysian company and business law.</p>"
            '<div class="eg">'
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    for msg in history:
        st.markdown(
            f'<div class="mu"><div class="b">{html.escape(msg.get("query", ""))}</div></div>',
            unsafe_allow_html=True,
        )
        qa_id = msg.get("qa_id") or _make_qa_id(msg.get("query", ""), msg.get("timestamp", ""))
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(msg.get("response", "*(no response recorded)*"))
            if msg.get("forms"):
                forms_from_history = [
                    FormEntry(
                        form_id="", form_number=f.get("form_number", ""), name=f["name"],
                        links=f.get("links") or (
                            [{"type": "pdf", "url": f["url"]}] if f.get("url") else []
                        ),
                        related_sections=[],
                        resource_type=f.get("resource_type", "form"),
                    )
                    for f in msg["forms"]
                ]

            cats = msg.get("categories_hit", [])
            if cats:
                badge_html = " ".join(
                    f'<span class="badge">{html.escape(c)}</span>' for c in cats
                )
                st.markdown(badge_html, unsafe_allow_html=True)
            _render_inline_feedback(msg, qa_id)


def _submit_feedback(
    msg: Dict,
    qa_id: str,
    rating: int,
    failure_key: Optional[str],
    comment: str,
    vote: str,          # "up" or "down"
) -> None:
    """Save feedback, optionally evict the cache entry, update session state."""
    fb_store = _feedback_store()
    store    = _store()
    rc       = _response_cache()

    fb_store.save(
        qa_id        = qa_id,
        query        = msg.get("query", ""),
        response     = msg.get("response", ""),
        citations    = msg.get("citations", []),
        rating       = rating,
        failure_type = failure_key,
        comment      = comment,
    )
    store.log_qa(
        msg.get("query", ""), msg.get("response", ""),
        msg.get("citations", []), rating, qa_id,
    )
    # Evict incorrect cached answer so next ask re-generates a fresh response
    if rating <= 2 and msg.get("cache_key"):
        rc.invalidate_entry(msg["cache_key"])

    st.session_state[f"fb_vote_{qa_id}"]     = vote
    st.session_state[f"fb_submitted_{qa_id}"] = True
    st.session_state.pop(f"fb_pending_{qa_id}", None)
    st.rerun()


def _render_inline_feedback(msg: Dict, qa_id: str) -> None:
    """
    Render thumbs-up / thumbs-down feedback bar inline inside a chat bubble.

    States:
      • Default   → 👍  👎  📥  (icon buttons, borderless)
      • 👎 clicked → failure-type dropdown + Submit / Skip
      • Submitted  → green or red confirmation pill
    """
    submitted_key = f"fb_submitted_{qa_id}"
    pending_key   = f"fb_pending_{qa_id}"

    st.markdown('<hr class="fb-hr">', unsafe_allow_html=True)

    # ── Already submitted ──────────────────────────────────────────────────
    if st.session_state.get(submitted_key):
        vote = st.session_state.get(f"fb_vote_{qa_id}", "up")
        if vote == "up":
            st.markdown(
                '<span class="fb-done fb-done-up">👍&nbsp; Helpful</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="fb-done fb-done-dn">👎&nbsp; Not helpful</span>',
                unsafe_allow_html=True,
            )
        return

    # ── Thumbs-down detail form ────────────────────────────────────────────
    if st.session_state.get(pending_key) == "negative_form":
        st.markdown("**What went wrong?** *(optional details help improve the system)*")

        failure_label = st.selectbox(
            "Failure type",
            options=list(FeedbackStore.FAILURE_TYPES.values()),
            key=f"fb_ft_{qa_id}",
            label_visibility="collapsed",
        )
        failure_key = next(
            k for k, v in FeedbackStore.FAILURE_TYPES.items() if v == failure_label
        )
        comment = st.text_area(
            "Details",
            placeholder="e.g. 'Section 17 was cited but the answer quoted Section 18'",
            key=f"fb_comment_{qa_id}",
            max_chars=3000,
            label_visibility="collapsed",
        )
        col_sub, col_skip, _ = st.columns([2, 2, 5])
        if col_sub.button("Submit", key=f"fb_sub_{qa_id}", type="primary"):
            _submit_feedback(msg, qa_id, 1, failure_key, comment, vote="down")
        if col_skip.button("Skip", key=f"fb_skip_{qa_id}"):
            _submit_feedback(msg, qa_id, 1, None, "", vote="down")
        return

    # ── Default: icon buttons row ──────────────────────────────────────────
    col_up, col_dn, col_dl, _ = st.columns([1, 1, 1, 10])

    if col_up.button("👍", key=f"fb_up_{qa_id}", help="Helpful"):
        _submit_feedback(msg, qa_id, 5, None, "", vote="up")

    if col_dn.button("👎", key=f"fb_dn_{qa_id}", help="Not helpful"):
        st.session_state[pending_key] = "negative_form"
        st.rerun()

    cits     = ", ".join(msg.get("citations", []))
    cats_str = ", ".join(msg.get("categories_hit", []))
    export   = (
        f"Q: {msg.get('query', '')}\n\n"
        f"A: {msg.get('response', '')}\n\n"
        f"Sources: {cits}\n"
        f"Categories: {cats_str}\n"
        f"Time: {msg.get('timestamp', '')}"
    )
    col_dl.download_button(
        "📥",
        data      = export,
        file_name = f"chatssm_{qa_id}.txt",
        mime      = "text/plain",
        help      = "Download this response",
        key       = f"fb_dl_{qa_id}",
    )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Check if indexes need building BEFORE rendering any UI ────────────────
    kb  = _kb()
    reg = _reg()

    enabled_sources = reg.all_enabled()
    needs_building = any(
        not os.path.exists(os.path.join(AppConfig.CACHE_DIR, f"{src.key}_index.pkl"))
        for src in enabled_sources
        if src.key not in kb._indexes or not kb._indexes[src.key].is_ready()
    )

    if needs_building:
        # Show a dedicated full-screen loading page — no sidebar, no header, no chat
        st.markdown(
            """
            <div style="
                display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:80vh; text-align:center;
            ">
                <h1 style="font-size:2rem; margin-bottom:8px;">⚖️ ChatSSM</h1>
                <p style="color:#666; margin-bottom:32px;">
                    Initializing knowledge base, please wait…
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress_bar = st.progress(0)
        status_text  = st.empty()

        for i, source in enumerate(enabled_sources):
            cache_path = os.path.join(AppConfig.CACHE_DIR, f"{source.key}_index.pkl")
            if os.path.exists(cache_path):
                progress_bar.progress((i + 1) / len(enabled_sources))
                continue
            status_text.markdown(f"**Building:** {source.name}  `({i+1}/{len(enabled_sources)})`")
            try:
                idx = kb.get_or_build(source)
                if idx and idx.is_ready():
                    status_text.markdown(f"✅ {source.name}")
                else:
                    status_text.markdown(f"⚠️ Skipped: {source.name} — run `python preprocess.py --key {source.key}`")
            except Exception as exc:
                logger.error("Error building %s: %s", source.key, exc)
                status_text.markdown(f"❌ Error: {source.name} — {exc}")
            progress_bar.progress((i + 1) / len(enabled_sources))

        status_text.markdown("✅ **Knowledge base ready! Reloading...**")
        CacheBuilder._warmup_llm()
        time.sleep(1)   # brief pause so user sees the done message
        st.rerun()      # rerun now renders the full chat UI cleanly
        return

    # ── All indexes ready — render normal chat UI ─────────────────────────────
    ollama_ok, selected_keys, cat_filter = _sidebar()
    _header()

    if not ollama_ok:
        st.error(
            "⚠️ **Ollama is not running.**  "
            "Open a terminal and run: `ollama serve`  then refresh this page."
        )
        st.stop()

    uid = _get_persistent_user_id()

    # MemoryManager requires the EmbeddingService singleton (_emb()), so it
    # cannot be created in _init_session() which runs before the singletons
    # are defined.  Create it here on first run, persist in session_state.
    if "mem_mgr" not in st.session_state:
        st.session_state["mem_mgr"] = MemoryManager(
            embedding_service = _emb(),
            session_id        = _get_persistent_user_id(),
        )

    if not st.session_state["chat_history"]:
        st.session_state["chat_history"] = _store().load_history(uid)

    _render_messages()

    # Auto-scroll
    just_submitted = st.session_state.pop("just_submitted", False)

    if just_submitted:
        # Post-submit: scroll immediately and keep following streaming content
        st.markdown(
            """
            <script>
                setTimeout(() => window.scrollTo(0, document.body.scrollHeight), 100);
                const _s = setInterval(() => window.scrollTo(0, document.body.scrollHeight), 500);
                setTimeout(() => clearInterval(_s), 4000);
            </script>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Page load: single scroll after content renders
        st.markdown(
            "<script>setTimeout(() => window.scrollTo(0, document.body.scrollHeight), 400);</script>",
            unsafe_allow_html=True,
        )
    
    prefill_val = st.session_state.pop("prefill", "")
    user_input = st.chat_input(
        "Ask anything...",
        key="chat_input",
        max_chars=2000,
    ) or prefill_val

    if st.session_state.pop("force_show_forms", False) and user_input:
        st.session_state["pending_form_query"] = user_input

    # Check if user confirmed a pending form request
    pending_q = st.session_state.pop("pending_form_query", None)
    if pending_q:
        # Treat as if user asked for forms directly for the pending query
        user_input = pending_q   # re-run with original query, forms will now show

    if user_input and user_input.strip():
        if not selected_keys:
            st.warning(
                "No active knowledge sources. "
                "Select at least one source in the sidebar."
            )
            st.stop()

        kb    = _kb()
        llm   = _llm()
        store = _store()
        opt   = _optimizer()
        memory: ConversationMemory = st.session_state["conv_memory"]
        lang = _detect_language(user_input)

        # ── Short-circuit for greetings — no retrieval needed ─────────────────
        if _GREETING_RE.match(user_input.strip()):
            greeting_response = _GREETING_RESPONSES.get(lang, _GREETING_RESPONSES["en"])
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(greeting_response)
            # Save to history but do NOT cache — greetings are trivial
            timestamp = datetime.now().isoformat()
            qa_id     = _make_qa_id(user_input, timestamp)
            memory.add_turn(user_input, greeting_response)
            record = {
                "qa_id":          qa_id,
                "query":          user_input,
                "response":       greeting_response,
                "citations":      [],
                "categories_hit": [],
                "timestamp":      timestamp,
                "cache_key":      "",
            }
            st.session_state["chat_history"].append(record)
            store.save_history(st.session_state["chat_history"], uid)
            st.session_state["just_submitted"] = True
            st.rerun()
            return

        st.markdown(
            f'<div class="mu"><div class="b">{html.escape(user_input)}</div></div>',
            unsafe_allow_html=True,
        )

        with st.spinner("_Thinking..._"):
            search_query = _expand_query(memory.rewrite_query(user_input, llm))

            result = kb.search(
                query         = search_query,
                selected_keys = selected_keys,
                cat_filter    = cat_filter,
                lang          = lang,
            )
            # ── Build section frequency from retrieved chunks ──────────────
            # Computed once here; passed to agent as the retrieval signal.
            _freq: Dict[tuple, int] = _build_section_freq(result.get("chunks", []))
            # If chunks gave nothing, fall back to any section refs in the query
            if not _freq:
                _sec_q = re.compile(
                    r'\b(?:section|seksyen)\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?'
                    r'(?=\b|\s|\.|,|;|\)|\(|$)', re.IGNORECASE,
                )
                for m in _sec_q.finditer(user_input):
                    ref = (m.group(1).lower(), m.group(2).lower() if m.group(2) else None)
                    _freq[ref] = 1

            # Persist section context for multi-turn follow-ups
            if _freq:
                st.session_state["_last_sections"] = _freq

            # ── Intent agent: resolve forms ────────────────────────────────
            agent      = _intent_agent()
            conv_turns = st.session_state["conv_memory"]._turns
            resolution = agent.resolve(
                query               = user_input,
                retrieved_sections  = _freq,
                conversation_history= conv_turns,
                lang                = lang,
            )

            # Convert agent output → FormEntry objects for downstream rendering
            matched_forms: List[FormEntry] = [
                FormEntry(
                    form_id      = "",
                    form_number  = f.get("form_number", ""),
                    name         = f["name"],
                    links        = [{"type": f.get("resource_type", "pdf"), "url": f["url"]}]
                                   if f.get("url") else [],
                    related_sections = [],
                    resource_type    = f.get("resource_type", "form"),
                )
                for f in resolution.forms
            ]

            # ask_user: procedural intent detected but no form found with confidence
            ask_user_for_form = (
                not matched_forms
                and resolution.intent is not None
                and resolution.intent.is_actionable
                and resolution.intent.confidence > 0.0
            )
            
            rc = _response_cache()
            cached = rc.get(user_input, selected_keys, cat_filter, lang)
            patches  = opt.get_patches()

            if matched_forms:
                form_lines = "\n".join(
                    f"  • {f.name}" + (f" ({f.form_number})" if f.form_number else "")
                    for f in matched_forms
                )
                if lang == "ms":
                    form_hint = (
                        f"\nARahan BORANG (WAJIB IKUT):\n"
                        f"Borang berikut diperlukan untuk prosedur ini:\n{form_lines}\n"
                        f"Anda MESTI menyebut nama borang di atas dengan TEPAT seperti yang ditulis "
                        f"(jangan singkat, paraphrase, atau ubah nama) dalam satu ayat semula jadi "
                        f"di penghujung jawapan anda.\n"
                        f"Contoh: \"Anda perlu mengemukakan [nama borang tepat] kepada Pendaftar.\"\n"
                        f"Jangan nyatakan URL atau pautan.\n"
                    )
                else:
                    form_hint = (
                        f"\nFORM INSTRUCTION (MANDATORY):\n"
                        f"The following form(s) are required for this procedure:\n{form_lines}\n"
                        f"You MUST write the form name(s) EXACTLY as shown above — do NOT shorten, "
                        f"paraphrase, or rename them. Include the exact name in one natural sentence "
                        f"at the end of your answer.\n"
                        f"Example: \"You will need to submit the [exact form name] to the Registrar.\"\n"
                        f"Do NOT include any URL or link.\n"
                    )
            else:
                form_hint = ""

            

        response = (
                    "No relevant sections were found in the selected knowledge sources. "
                    "Please rephrase your question, enable more sources, or consult a "
                    "licensed professional."
                )

        if cached:
            response = cached["response"]
            # Inject hyperlinks into the cached response (links are already stored
            # in the response, but re-inject from matched_forms in case the cache
            # was saved before inject was applied in earlier sessions).
            if matched_forms:
                response = _inject_form_links(response, matched_forms, lang=lang)
            with st.chat_message("assistant", avatar="⚖️"):
                stream_box = st.empty()
                words = response.split(" ")
                buf = ""
                for i, word in enumerate(words):
                    buf += ("" if i == 0 else " ") + word
                    if i % 8 == 7:
                        stream_box.markdown(buf + "▌")
                        time.sleep(0.04)
                stream_box.markdown(response)
                if ask_user_for_form:
                    msg = (
                        "Berdasarkan soalan anda, prosedur ini mungkin memerlukan borang. "
                        "Adakah anda ingin saya paparkan borang yang berkaitan?"
                        if lang == "ms" else
                        "Based on your question, this process may require a form. "
                        "Would you like me to show the relevant form?"
                    )
                    st.info(msg, icon="📋")
                    col_yes, col_no, _ = st.columns([2, 2, 6])
                    if col_yes.button("✅ Yes, show form", key=f"form_yes_{qa_id}"):
                        st.session_state["prefill"] = user_input
                        st.session_state["force_show_forms"] = True
                        st.rerun()
                    if col_no.button("❌ No thanks", key=f"form_no_{qa_id}"):
                        st.session_state.pop("pending_form_query", None)
        else:
            if not result["chunks"]:
                with st.chat_message("assistant", avatar="⚖️"):
                    st.markdown(response)

            else:
                system = llm._SYSTEM_BASE
                if patches:
                    system += f"\n{patches}\n"
                act_hint = (
                    f"\nNOTE: This query relates to the {result.get('detected_act')}. "
                    f"Prioritize chunks from that Act when constructing your answer.\n"
                ) if result.get("detected_act") else ""

                context_parts = []
                chunk_types_list = result.get("chunk_types", [])

                for i, chunk in enumerate(result["chunks"]):
                    src = result["chunk_sources"][i] if i < len(result["chunk_sources"]) else f"Source {i+1}"
                    ctype = chunk_types_list[i] if i < len(chunk_types_list) else "text"
                    type_label = " | VISUAL CONTENT" if ctype == "visual" else ""
                    if len(chunk) > AppConfig.MAX_CHUNK_CHARS:
                        if "[TABLE]" in chunk:
                            # Find the last complete table row before the limit
                            cut = chunk.rfind("\n", 0, AppConfig.MAX_TABLE_CHARS)
                            safe_chunk = chunk[:cut] if cut > 0 else chunk[:AppConfig.MAX_TABLE_CHARS]
                        else:
                            cut = AppConfig.MAX_CHUNK_CHARS
                            for m in re.finditer(r'(?<=[.!?])\s', chunk[:AppConfig.MAX_CHUNK_CHARS]):
                                cut = m.end()
                            safe_chunk = chunk[:cut] if cut > 60 else chunk[:AppConfig.MAX_CHUNK_CHARS]
                    else:
                        safe_chunk = chunk
                    context_parts.append(f"[SOURCE: {src}{type_label}]\n{safe_chunk}\n[/SOURCE: {src}]")
                context_block = "\n\n".join(context_parts)
                history_block = memory.build_history_block()
                # Build memory block using the query embedding.
                # _emb().embed() hits the in-memory cache (populated by kb.search
                # moments earlier) so this costs ~0ms — no extra Ollama call.
                _query_vec = _emb().embed(search_query)
                memory_block = st.session_state["mem_mgr"].build_memory_block(
                    query_vec=_query_vec,
                    query_text=search_query, 
                )

                if lang == "ms":
                    lang_directive = "⚠️ WAJIB: Jawab SEPENUHNYA dalam Bahasa Malaysia. Jangan gunakan bahasa Inggeris langsung.\n"
                elif lang == "mixed":
                    lang_directive = "NOTE: User mixed Malay and English. Respond in English, Malay legal terms are acceptable.\n"
                else:
                    lang_directive = ""

                prompt = (
                    f"/no_think\n{history_block}{memory_block}<<CONTEXT_BLOCK>>\n{'─'*72}\n{context_block}\n{'─'*72}\n\n"
                    f"<<USER_QUESTION>>\n{user_input}\n\n{act_hint}{lang_directive}{form_hint}"
                    f"Before writing, identify the single most relevant section in the CONTEXT above.\n"
                    f"ANSWER (explain conversationally, cite every fact with its source, stay within CONTEXT):\n"
                )

                def _stream_with_retry(p: str, s: str, temp: Optional[float] = None):
                    """Run one LLM call; return (generator, first_token_or_None)."""
                    gen   = llm._call(p, s, temp_override=temp)
                    first = next(gen, None)
                    return gen, first
                
                raw_tokens  = []

                with st.spinner("_Generating response..._"):
                    try:
                        gen, first_token = _stream_with_retry(prompt, system)
                    except Exception as exc:
                        gen, first_token = iter([]), f"❌ Error during generation: {exc}"
                        logger.error("LLM call failed before first token: %s", exc, exc_info=True)

                with st.chat_message("assistant", avatar="⚖️"):
                    stream_box = st.empty()
                    try:
                        if first_token:
                            raw_tokens.append(first_token)
                            stream_box.markdown("".join(raw_tokens) + "▌")

                        for token in gen:
                            raw_tokens.append(token)
                            stream_box.markdown("".join(raw_tokens) + "▌")

                    except Exception as exc:
                        raw_tokens.append(f"\n\n❌ Error during streaming: {exc}")
                        logger.error("Streaming error in main(): %s", exc, exc_info=True)

                    raw = "".join(raw_tokens)

                    truncated = (
                        raw.strip()
                        and not re.search(r'[.!?)\]`*|\-\w]\s*$', raw, re.IGNORECASE)
                        and not raw.startswith("❌")
                    )
                    if (len(raw.strip()) < 30 or truncated) and not raw.startswith("❌"):
                        logger.warning("LLM response too short or truncated; retrying once.")
                        stream_box.markdown("_Response was too short, retrying once..._")
                        raw_tokens = []

                        with st.spinner("_Retrying generation..._"):
                            try:
                                gen2, first2 = _stream_with_retry(prompt, system, temp=0.15)
                            except Exception as exc:
                                gen2 = iter([])
                                first2 = None
                                logger.error("LLM retry failed: %s", exc, exc_info=True)
                        
                        if first2 and not str(first2).startswith("❌"):
                            stream_box.empty()
                            raw_tokens.append(first2)
                            stream_box.markdown("".join(raw_tokens) + "▌")
                            for token in gen2:
                                raw_tokens.append(token)
                                stream_box.markdown("".join(raw_tokens) + "▌")
                            raw = "".join(raw_tokens)
                        else:
                            # Retry also failed — show a clear, helpful message
                            response = (
                                "I wasn't able to generate a response for this question. "
                                "This can happen with complex or ambiguous queries. "
                                "Please try rephrasing your question, or break it into smaller parts."
                            )
                            raw = ""
                            stream_box.markdown(response)

                    if raw.strip():
                        response = llm._postprocess(raw, lang=lang)
                        if matched_forms:
                            response = _inject_form_links(response, matched_forms, lang=lang)
                        if not _OOS_PATTERN.search(response):
                            rc.put(user_input, selected_keys, cat_filter,
                                response, result.get("citations", []), result.get("categories_hit", []), lang)
                        stream_box.markdown(response)
                    elif ask_user_for_form:
                        confirm_msg = (
                            "Berdasarkan soalan anda, prosedur ini mungkin memerlukan borang. "
                            "Adakah anda ingin saya paparkan borang yang berkaitan?"
                            if lang == "ms" else
                            "Based on your question, this process may require a form to be submitted. "
                            "Would you like me to show the relevant form?"
                        )
                        st.info(confirm_msg, icon="📋")
                        col_yes, col_no, _ = st.columns([2, 2, 6])
                        if col_yes.button("✅ Yes, show form", key=f"form_yes_{qa_id}"):
                            st.session_state["prefill"] = user_input
                            st.session_state["force_show_forms"] = True
                            st.rerun()
                        if col_no.button("❌ No thanks", key=f"form_no_{qa_id}"):
                            st.session_state.pop("pending_form_query", None)
                            st.session_state.pop("force_show_forms", None)

        timestamp = datetime.now().isoformat()
        qa_id     = _make_qa_id(user_input, timestamp)
        memory.add_turn(user_input, response)
        # Inspect the user's message for memory-worthy content (preferences,
        # corrections, instructions).  Never called on the LLM response to
        # prevent hallucinations from being stored.
        st.session_state["mem_mgr"].observe(user_input, lang=lang)

        record = {
            "qa_id":          qa_id,
            "query":          user_input,
            "response":       response,
            "lang":           lang,
            "citations":      result.get("citations", []),
            "categories_hit": result.get("categories_hit", []),
            "timestamp":      timestamp,
            "cache_key":      ResponseCache._make_key(user_input, selected_keys, cat_filter, lang),
            "forms": [
                {
                    "form_number": f.form_number,
                    "name":        f.name,
                    "resource_type": f.resource_type,
                    "links": [l for l in f.links if l.get("type") in ("pdf", "portal", "platform")],
                }
                for f in matched_forms
            ],
        }
        st.session_state["chat_history"].append(record)
        store.save_history(st.session_state["chat_history"], uid)
        st.session_state["just_submitted"] = True
        st.rerun()

if __name__ == "__main__":
    main()