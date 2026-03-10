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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple
from filelock import FileLock

import numpy as np
import pandas as pd
import requests
import streamlit as st
import html
import tempfile, shutil
import csv

from chunk import Chunk, SearchResult

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
    LLM_MAX_TOKENS:  int   = 1200
    LLM_TIMEOUT:     int   = 300    # seconds
    LLM_NUM_CTX: int = 6144   #Ollama default is 2048; set higher to fit system prompt + long contexts without truncation. Must be >= LLM_MAX_TOKENS + max context chunk size.
    LLM_NUM_GPU:       int = int(os.environ.get("CHATSSM_LLM_NUM_GPU",       "-1"))

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_TIMEOUT: int = 15     # seconds per call
    EMBEDDING_WORKERS: int = 4      # Parallel workers for index building.
    EMBEDDING_NUM_GPU: int = int(os.environ.get("CHATSSM_EMBEDDING_NUM_GPU", "-1"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.45
    TOP_K_PER_SOURCE:     int   = 4
    GLOBAL_TOP_K:         int   = 8   # total chunks sent to LLM
    MAX_CHUNK_CHARS:     int   = 1200  # Truncate chunks in the prompt to this many characters to avoid hitting LLM token limits.
    MAX_TABLE_CHARS:  int = 2500

    # ── Paths ─────────────────────────────────────────────────────────────────
    SOURCES_CONFIG:  str = "knowledge_sources.json"
    SOURCES_DIR:     str = os.path.join("knowledge_base", "sources")
    PROCESSED_DIR:   str = os.path.join("knowledge_base", "processed")
    CACHE_DIR:       str = os.path.join("knowledge_base", "embeddings")
    DATA_DIR:        str = "qa_data"

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in [cls.SOURCES_DIR, cls.PROCESSED_DIR, cls.CACHE_DIR, cls.DATA_DIR]:
            try:
                os.makedirs(d, exist_ok=True)
            except OSError as exc:
                logger.warning("Could not create directory '%s': %s", d, exc)


AppConfig.ensure_dirs()

_CHAT_HISTORY_FILE = os.path.join(AppConfig.DATA_DIR, "chat_history.json")
_QA_LOG_FILE       = os.path.join(AppConfig.DATA_DIR, "qa_log.csv")
_FEEDBACK_FILE     = os.path.join(AppConfig.DATA_DIR, "feedback.json")
_EMBEDDING_CACHE   = os.path.join(AppConfig.CACHE_DIR, "embedding_cache.pkl")

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

    # ── Readiness ─────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._matrix is not None and bool(self._chunks)

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
            raw.append(Chunk(
                text             = chunk_text,
                source_key       = self._key,
                source_name      = self._name,
                category         = self._cat,
                section          = section,
                part             = part,
                relates_to_acts  = src_entry.relates_to_acts if src_entry else [],
            ))

        return self._embed_and_build(raw)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray) -> List[SearchResult]:
        if self._matrix is None or not self._chunks:
            return []

        scores: np.ndarray = self._matrix @ query_vec   # shape (N,)

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
                pickle.dump({"version": 3, "chunks": self._chunks, "matrix": self._matrix}, fh)
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

        texts = [chunk.text for chunk in raw]
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
            def _word_match(kw: str, text: str) -> bool:
                return bool(re.search(r'\b' + re.escape(kw) + r'\b', text))
            
            matches = sum(1 for kw in keywords if _word_match(kw, query_lower))
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
                hits = [(r, source) for r in idx.search(query_vec)]
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
            with ThreadPoolExecutor(max_workers=_SEARCH_WORKERS) as pool:
                for key, hits in pool.map(_search_one, sources_to_search):
                    if hits:
                        hits_by_source[key] = hits

        all_hits: List[Tuple[SearchResult, SourceEntry]] = []
        for source_results in hits_by_source.values():
            all_hits.extend(source_results[:AppConfig.TOP_K_PER_SOURCE])

        all_hits.sort(key=lambda x: x[0].score, reverse=True)

        seen:          set                = set()
        chunks:        List[str]          = []
        chunk_sources: List[str]          = []
        results:       List[SearchResult] = []
        citations:     List[str]          = []
        cats_hit:      List[str]          = []

        for result, source in all_hits:
            if result.chunk.text in seen:
                continue
            seen.add(result.chunk.text)
            chunks.append(result.chunk.text)
            chunk_sources.append(source.name)
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
        "chunks": [], "chunk_sources": [], "results": [],
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
    MAX_HIST_TOKENS:  int = 400   # hard token budget for history block
    REWRITE_THRESHOLD: int = 8    # queries shorter than this word count get rewritten

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
        if len(user_input.split()) > self.REWRITE_THRESHOLD:
            return user_input   # Long queries are self-contained
        if not self._turns:
            return user_input   # No history to reference

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
            if rewritten and 3 < len(rewritten) < 200:
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
            "Use ONLY to resolve references ('it', 'that', 'the fee', etc.).\n"
            "Do NOT answer from history. Answer only from CONTEXT below.\n"
            "─" * 40 + "\n"
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
        # First sentence of response (the direct answer)
        first_sent = response.split(".")[0].strip()[:150]

        facts = sections[:3] + amounts[:2] + deadlines[:2]
        summary = first_sent
        if facts:
            summary += f" [{', '.join(facts)}]"
        return summary[:300]

# =============================================================================
# LLM SERVICE
# =============================================================================


class LLMService:
    """
    Builds the prompt and calls Ollama.
    Accepts dynamic prompt patches from PromptOptimizer.
    """

    _SYSTEM_BASE = """\
You are a knowledgeable and approachable legal assistant for Suruhanjaya Syarikat Malaysia (SSM).
Your role is to help users understand Malaysian company law in a clear, conversational way —
as if a senior corporate secretary is explaining it to a colleague over a desk.

ACCURACY RULES  —  NON-NEGOTIABLE

RULE 1 — STAY WITHIN THE CONTEXT
  • Every factual statement MUST come from the CONTEXT block below.
  • Do NOT add information from your general training knowledge.
  • If the answer is not in the CONTEXT, say so clearly (see Rule 5).

RULE 2 — CITE EVERYTHING
  • After every factual statement, cite the source in bold.
  • Format: **(Section 14(1), Companies Act 2016)** or **(Practice Note 3/2018, para 8)**
  • Never make a factual claim without a citation immediately after it.

RULE 3 — EXACT NUMBERS, DATES AND THRESHOLDS
  • Copy every number, deadline, ringgit amount, and percentage exactly as it appears
    in the CONTEXT. Do not round, approximate, or paraphrase these.

RULE 4 — LEGAL TERMS
  • Reproduce defined legal terms (e.g. "beneficial owner", "lodgement") exactly as written.
  • Paraphrasing legal definitions changes their legal meaning — never do this.

RULE 5 — OUT OF SCOPE
  • If the CONTEXT does not contain enough information to answer the question, respond:
    "I couldn't find this in the documents I have access to. For this specific question,
    I'd recommend checking directly with SSM or consulting a licensed company secretary."
  • Never speculate or infer beyond what is explicitly stated.

TONE AND FORMAT RULES

TONE — Write in a warm, natural, professional tone:
  • Use plain English where possible, then introduce the legal term.
    Example: "The deadline to file (called the 'lodgement period') is 30 days..."
  • Address the user directly using "you" and "your company".
  • Avoid sounding like a legal statute. Explain it, then cite it.

FORMAT — Choose the structure that fits the answer:
  • Short factual questions → 1-5 paragraph prose answer with inline citations.
  • Step-by-step procedures → numbered list, one step per item.
  • Comparing two things   → short table or two labelled paragraphs.
  • Do NOT use bullet points, headers, or bold text just to look structured.
    Use them only when they genuinely make the answer clearer.

OPENING — Start with a direct answer to the question, not a preamble.
  • Good: "Yes, a private company must file its annual return within 30 days..."
  • Bad:  "According to the CONTEXT provided, I will now explain the requirements..."

CLOSING — End naturally. Do not add disclaimers like "please consult a lawyer" 
  unless the question is genuinely outside the documents (Rule 5).
"""

    def generate(self, query, context_chunks, citations=None, prompt_patches="", detected_act=None):
        if not context_chunks:
            return (
                "No relevant sections were found in the selected knowledge sources. "
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
    def _postprocess(text: str) -> str:
        if (text.startswith("❌") or text.startswith("No relevant sections") or text.startswith("This information is not found")):
            return text

        # Strip reasoning blocks
        text = re.sub(
            r"<(?:think|thinking|reasoning|reflection)>.*$",
            "", text, flags=re.DOTALL | re.IGNORECASE
        )

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        # Detect truncation
        if text and not re.search(r"[.!?)\]`*|\-\w]\s*$", text, re.IGNORECASE):
            text += (
                "\n\n---\n"
                "⚠️ *The response was cut off before completion. "
                "Try asking a more specific question, or break it into smaller parts.*"
            )

        return text

    def _call(self, prompt: str, system: str="") -> Generator[str, None, None]:
        try:
            r = requests.post(
                f"{AppConfig.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model":  AppConfig.LLM_MODEL, 
                    "prompt": prompt,
                    "system": system,
                    "stream": True,
                    "options": {
                        "temperature": AppConfig.LLM_TEMPERATURE,
                        "top_p":       AppConfig.LLM_TOP_P,
                        "top_k":       AppConfig.LLM_TOP_K,
                        "num_predict": AppConfig.LLM_MAX_TOKENS,
                        "num_ctx":     AppConfig.LLM_NUM_CTX,
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
            st.markdown("### ⏳ Initializing Knowledge Base (First Time Only)")
            st.info("This takes 1–2 minutes on first run, then everything is cached.")

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

        return True


# =============================================================================
# STORAGE SERVICE
# =============================================================================

_qa_log_lock = threading.Lock()

class StorageService:
    """Thin I/O layer: chat history (JSON) + Q&A log (CSV)."""

    @staticmethod
    def load_history() -> List[Dict]:
        try:
            if os.path.exists(_CHAT_HISTORY_FILE):
                with open(_CHAT_HISTORY_FILE, "r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception as exc:
            logger.error("Load history failed: %s", exc)
        return []

    @staticmethod
    def save_history(history: List[Dict]) -> None:
        tmp_path = _CHAT_HISTORY_FILE + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                history_to_save = history[-200:]
                json.dump(history_to_save, fh, indent=2, default=str, ensure_ascii=False)
            shutil.move(tmp_path, _CHAT_HISTORY_FILE)
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

.fb-submitted {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #166534;
    margin-top: 6px;
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

    with st.sidebar:
        st.markdown("## ⚖️ ChatSSM")

        if st.button("➕ New Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            store.save_history([])
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
            expanded = (cat == "Legislations")

            with st.expander(f"{icon} {cat}  ({len(srcs)})", expanded=expanded):
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


def _render_messages() -> None:
    history = st.session_state.get("chat_history", [])
    if not history:
        st.markdown(
            '<div class="welcome">'
            "<h2>Welcome to ChatSSM</h2>"
            "<p>Ask any question about Malaysian company and business law.</p>"
            '<div class="eg">'
            "Examples:<br>"
            "• What are the qualifications to become a company director?<br>"
            "• When must a company hold its Annual General Meeting?<br>"
            "• What are the penalties for late filing of annual returns?<br>"
            "• How do I register a sole proprietorship?<br>"
            "• What does the LLP Act say about winding up?<br>"
            "• What are the requirements to strike off a company?"
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    for msg in history:
        st.markdown(
            f'<div class="mu"><div class="b">{html.escape(msg.get("query", ""))}</div></div>',
            unsafe_allow_html=True,
        )
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(msg.get("response", "*(no response recorded)*"))
            cats = msg.get("categories_hit", [])
            if cats:
                badge_html = " ".join(
                    f'<span class="badge">{html.escape(c)}</span>' for c in cats
                )
                st.markdown(badge_html, unsafe_allow_html=True)


def _render_feedback(latest: Dict) -> None:
    """
    Render the feedback panel for the most recent Q&A exchange.

    Features:
    - 👍 / 😐 / 👎 quick rating buttons
    - On 👎: shows failure-type dropdown + optional comment box
    - Double-submit guard via session state key per qa_id
    - Download button for the response text
    """
    fb_store = _feedback_store()
    store    = _store()

    qa_id     = latest.get("qa_id", "unknown")
    submitted_key = f"fb_submitted_{qa_id}"
    pending_key   = f"fb_pending_{qa_id}"

    with st.expander("📋 Rate this response & Export", expanded=False):

        # ── Already submitted guard ────────────────────────────────────────
        if st.session_state.get(submitted_key):
            st.markdown(
                '<div class="fb-submitted">✅ Feedback submitted — thank you!</div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.get(pending_key) == "negative_form":
            # ── Negative feedback form ─────────────────────────────────────
            st.markdown("**What went wrong?** *(optional details help improve the system)*")

            failure_label = st.selectbox(
                "Failure type",
                options=list(FeedbackStore.FAILURE_TYPES.values()),
                key=f"fb_ft_{qa_id}",
            )
            # Map label back to key
            failure_key = next(
                k for k, v in FeedbackStore.FAILURE_TYPES.items()
                if v == failure_label
            )

            comment = st.text_area(
                "Additional details (optional)",
                placeholder="e.g. 'Section 17 was cited but the answer quoted Section 18'",
                key=f"fb_comment_{qa_id}",
                max_chars=3000,
            )

            col_sub, col_skip = st.columns(2)
            if col_sub.button("📤 Submit feedback", use_container_width=True, key=f"fb_sub_{qa_id}"):
                fb_store.save(
                    qa_id        = qa_id,
                    query        = latest.get("query", ""),
                    response     = latest.get("response", ""),
                    citations    = latest.get("citations", []),
                    rating       = 1,
                    failure_type = failure_key,
                    comment      = comment,
                )
                store.log_qa(
                    latest.get("query", ""), latest.get("response", ""),
                    latest.get("citations", []), 1, qa_id,
                )
                st.session_state[submitted_key] = True
                st.session_state.pop(pending_key, None)
                st.rerun()

            if col_skip.button("⏭ Skip details", use_container_width=True, key=f"fb_skip_{qa_id}"):
                fb_store.save(
                    qa_id     = qa_id,
                    query     = latest.get("query", ""),
                    response  = latest.get("response", ""),
                    citations = latest.get("citations", []),
                    rating    = 1,
                )
                store.log_qa(
                    latest.get("query", ""), latest.get("response", ""),
                    latest.get("citations", []), 1, qa_id,
                )
                st.session_state[submitted_key] = True
                st.session_state.pop(pending_key, None)
                st.rerun()

        else:
            # ── Initial rating buttons ─────────────────────────────────────
            st.markdown("**Was this response helpful?**")
            c1, c2, c3 = st.columns(3)

            if c1.button("👍 Helpful", use_container_width=True, key=f"fb_pos_{qa_id}"):
                fb_store.save(
                    qa_id     = qa_id,
                    query     = latest.get("query", ""),
                    response  = latest.get("response", ""),
                    citations = latest.get("citations", []),
                    rating    = 5,
                )
                store.log_qa(
                    latest.get("query", ""), latest.get("response", ""),
                    latest.get("citations", []), 5, qa_id,
                )
                st.session_state[submitted_key] = True
                st.rerun()

            if c2.button("😐 Okay", use_container_width=True, key=f"fb_neu_{qa_id}"):
                fb_store.save(
                    qa_id     = qa_id,
                    query     = latest.get("query", ""),
                    response  = latest.get("response", ""),
                    citations = latest.get("citations", []),
                    rating    = 3,
                )
                store.log_qa(
                    latest.get("query", ""), latest.get("response", ""),
                    latest.get("citations", []), 3, qa_id,
                )
                st.session_state[submitted_key] = True
                st.rerun()

            if c3.button("👎 Needs work", use_container_width=True, key=f"fb_neg_{qa_id}"):
                # Don't submit yet — open the detail form
                st.session_state[pending_key] = "negative_form"
                st.rerun()

        # ── Download ───────────────────────────────────────────────────────
        st.markdown("---")
        cits = ", ".join(latest.get("citations", []))
        cats = ", ".join(latest.get("categories_hit", []))
        export = (
            f"Q: {latest['query']}\n\n"
            f"A: {latest['response']}\n\n"
            f"Sources: {cits}\n"
            f"Categories searched: {cats}\n"
            f"Time: {latest['timestamp']}"
        )
        st.download_button(
            "📥 Download response",
            data=export,
            file_name=f"chatssm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            use_container_width=True,
        )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    ollama_ok, selected_keys, cat_filter = _sidebar()
    _header()

    if not ollama_ok:
        st.error(
            "⚠️ **Ollama is not running.**  "
            "Open a terminal and run: `ollama serve`  then refresh this page."
        )
        st.stop()

    kb  = _kb()
    reg = _reg()
    if not CacheBuilder.ensure_indexes_ready(kb, reg):
        st.warning("Could not initialize knowledge base. Please check logs.")
        st.stop()

    if not st.session_state["chat_history"]:
        st.session_state["chat_history"] = _store().load_history()

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
        "Ask about legislation, practice notes, guidelines, circulars, FAQ, or forms…",
        key="chat_input",
    ) or prefill_val

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

        st.markdown(
            f'<div class="mu"><div class="b">{html.escape(user_input)}</div></div>',
            unsafe_allow_html=True,
        )

        memory: ConversationMemory = st.session_state["conv_memory"]
        with st.spinner("_Thinking..._"):
            search_query = memory.rewrite_query(user_input, llm)
            result = kb.search(
                query         = search_query,
                selected_keys = selected_keys,
                cat_filter    = cat_filter,
            )
            patches  = opt.get_patches()

        response = (
                    "No relevant sections were found in the selected knowledge sources. "
                    "Please rephrase your question, enable more sources, or consult a "
                    "licensed professional."
                )

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

            for i, chunk in enumerate(result["chunks"]):
                src = result["chunk_sources"][i] if i < len(result["chunk_sources"]) else f"Source {i+1}"
                if len(chunk) > AppConfig.MAX_CHUNK_CHARS:
                    if "[TABLE]" in chunk:
                        # Find the last complete table row before the limit
                        cut = chunk.rfind("\n", 0, AppConfig.MAX_TABLE_CHARS)
                        safe_chunk = chunk[:cut] if cut > 0 else chunk[:AppConfig.MAX_TABLE_CHARS]
                    else:
                        safe_chunk = chunk[:AppConfig.MAX_CHUNK_CHARS]
                else:
                    safe_chunk = chunk
                context_parts.append(f"[SOURCE: {src}]\n{safe_chunk}\n[/SOURCE: {src}]")
            context_block = "\n\n".join(context_parts)
            history_block = memory.build_history_block()
            prompt = (
                f"/no_think\n{history_block}<<CONTEXT_BLOCK>>\n{'─'*72}\n{context_block}\n{'─'*72}\n\n"
                f"<<USER_QUESTION>>\n{user_input}\n\n{act_hint}\n"
                f"ANSWER (explain conversationally, cite every fact with its source, stay within CONTEXT):\n"
            )

            def _stream_with_retry(p: str, s: str):
                """Run one LLM call; return (generator, first_token_or_None)."""
                gen   = llm._call(p, s)
                first = next(gen, None)   # blocks here — spinner shows during this wait
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

                if len(raw.strip()) < 30 and not raw.startswith("❌"):
                    logger.warning("LLM returned near-empty response; retrying once.")
                    stream_box.markdown("_Response was too short, retrying once..._")
                    raw_tokens = []

                    with st.spinner("_Retrying generation..._"):
                        try:
                            gen2, first2 = _stream_with_retry(prompt, system)
                        except Exception as exc:
                            gen2, first2 = iter([]), f"❌ Retry failed: {exc}"
                            logger.error("LLM retry failed: %s", exc, exc_info=True)
                    stream_box.empty()
                    
                    if first2:
                        raw_tokens.append(first2)
                        stream_box.markdown("".join(raw_tokens) + "▌")
                    for token in gen2:
                        raw_tokens.append(token)
                        stream_box.markdown("".join(raw_tokens) + "▌")
                    raw = "".join(raw_tokens)

                response = llm._postprocess(raw)
                stream_box.markdown(response)  # Finalize the response display (remove the "▌" cursor)

        timestamp = datetime.now().isoformat()
        qa_id     = _make_qa_id(user_input, timestamp)
        memory.add_turn(user_input, response)

        record = {
            "qa_id":          qa_id,
            "query":          user_input,
            "response":       response,
            "citations":      result.get("citations", []),
            "categories_hit": result.get("categories_hit", []),
            "timestamp":      timestamp,
        }
        st.session_state["chat_history"].append(record)
        store.save_history(st.session_state["chat_history"])
        st.session_state["just_submitted"] = True
        st.rerun()

    # Feedback panel for most recent exchange
    if st.session_state["chat_history"]:
        _render_feedback(st.session_state["chat_history"][-1])

if __name__ == "__main__":
    main()