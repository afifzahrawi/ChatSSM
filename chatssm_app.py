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
  5. Pull models:    ollama pull qwen3-embedding:latest
                     ollama pull deepseek-r1:8b
  6. Start app:      streamlit run chatssm_app.py

Architecture
------------
  SourceRegistry   - reads knowledge_sources.json
  EmbeddingService - two-level cache (RAM + disk), numpy-normalised vectors
  DocumentIndex    - per-source numpy matrix; O(1) vectorised cosine search
  KnowledgeBase    - orchestrates all indexes; merges and ranks results
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

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
    EMBEDDING_MODEL: str   = "qwen3-embedding:latest"
    LLM_MODEL:       str   = "deepseek-r1:8b"

    # ── LLM sampling (deterministic = no hallucination drift) ─────────────────
    LLM_TEMPERATURE: float = 0.0
    LLM_TOP_P:       float = 0.1
    LLM_TOP_K:       int   = 10
    LLM_MAX_TOKENS:  int   = 3000  
    LLM_TIMEOUT:     int   = 300    # seconds

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_TIMEOUT: int = 60     # seconds per call
    EMBEDDING_WORKERS: int = 4      # Parallel workers for index building.

    # ── Retrieval ─────────────────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.35
    TOP_K_PER_SOURCE:     int   = 6
    GLOBAL_TOP_K:         int   = 12   # total chunks sent to LLM

    # ── Paths ─────────────────────────────────────────────────────────────────
    SOURCES_CONFIG:  str = "knowledge_sources.json"
    SOURCES_DIR:     str = os.path.join("knowledge_base", "sources")   # auto-discovery root
    PROCESSED_DIR:   str = os.path.join("knowledge_base", "processed")
    CACHE_DIR:       str = os.path.join("knowledge_base", "embeddings")
    DATA_DIR:        str = "qa_data"

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in [cls.SOURCES_DIR, cls.PROCESSED_DIR, cls.CACHE_DIR, cls.DATA_DIR]:
            os.makedirs(d, exist_ok=True)


AppConfig.ensure_dirs()

_CHAT_HISTORY_FILE = os.path.join(AppConfig.DATA_DIR, "chat_history.json")
_QA_LOG_FILE       = os.path.join(AppConfig.DATA_DIR, "qa_log.csv")
_EMBEDDING_CACHE   = os.path.join(AppConfig.CACHE_DIR, "embedding_cache.pkl")

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SourceEntry:
    key:        str
    name:       str
    category:   str
    type:       str           # "pdf" or "csv"
    enabled:    bool = True
    url:        Optional[str] = None
    local_path: Optional[str] = None

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
_VALID_DOC_TYPES: List[str] = ["act", "general", "faq"]


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
                    if not fname.lower().endswith((".pdf", ".csv")):
                        continue
                    fpath = os.path.join(dt_path, fname)
                    key   = _filename_to_key(fname)
                    if key in discovered:
                        continue
                    discovered[key] = SourceEntry(
                        key        = key,
                        name       = os.path.splitext(fname)[0],
                        category   = cat_name,
                        type       = "csv" if fname.lower().endswith(".csv") else "pdf",
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
        for raw in data.get("sources", []):
            if "_note" in raw or "_comment" in raw:
                continue
            try:
                key = raw["key"]
                overrides[key] = SourceEntry(
                    key        = key,
                    name       = raw.get("name", key),
                    category   = raw.get("category", ""),
                    type       = raw.get("type", "pdf"),
                    enabled    = raw.get("enabled", True),
                    url        = raw.get("url"),
                    local_path = raw.get("local_path"),
                )
            except KeyError:
                pass
        return overrides

    def _load(self) -> None:
        folder  = self._scan_folders()
        json_ov = self._load_json()

        merged: Dict[str, SourceEntry] = dict(folder)

        for key, je in json_ov.items():
            if key in merged:
                base = merged[key]
                merged[key] = SourceEntry(
                    key        = key,
                    name       = je.name if je.name != key else base.name,
                    category   = je.category or base.category,
                    type       = je.type,
                    enabled    = je.enabled,
                    url        = je.url or base.url,
                    local_path = je.local_path or base.local_path,
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
    s = st.session_state.get("cache_stats", {"hits": 0, "misses": 0})
    s["hits" if hit else "misses"] += 1
    st.session_state["cache_stats"] = s


class EmbeddingService:
    """
    Embeddings are unit-normalised so cosine_similarity == dot_product.

    Two-level cache:
      L1 - in-process dict  (instant, lives for the current server process)
      L2 - pickle on disk   (survives app restarts)

    Thread safety:
      _disk_lock  - serialises all writes to the pickle file
      _mem_lock   - serialises reads/writes to the shared in-memory dict

    Parallel embedding (embed_batch):
      Sends N requests to Ollama simultaneously using a ThreadPoolExecutor.
      Because each request is pure I/O (waiting for Ollama to respond),
      Python's GIL is not a bottleneck. With 4 workers you typically get
      3–4× speedup on a CPU-only machine and 6–8× on a GPU-accelerated one.
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
            if key in self._disk:
                _bump(hit=True)
                self._mem[key] = self._disk[key]
                return self._disk[key]

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

        How it works:
            ThreadPoolExecutor submits all requests at once. Ollama queues them
            internally. Because the bottleneck is GPU/CPU compute inside Ollama
            (not Python), more workers = more throughput up to Ollama's concurrency
            limit (usually equal to your GPU count).
        """
        total   = len(texts)
        results: Dict[int, Optional[np.ndarray]] = {}

        # Check cache first — avoid sending already-cached texts to Ollama at all
        to_fetch: Dict[int, str] = {}      # index → text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = None
                continue
            key = hashlib.md5(text.encode()).hexdigest()
            with self._mem_lock:
                if key in self._mem:
                    _bump(hit=True)
                    results[i] = self._mem[key]
                    continue
                if key in self._disk:
                    _bump(hit=True)
                    vec = self._disk[key]
                    self._mem[key] = vec
                    results[i] = vec
                    continue
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
        completed = total - len(to_fetch)   # already satisfied from cache

        def _fetch(idx_text: Tuple[int, str]) -> Tuple[int, Optional[np.ndarray]]:
            # NOTE: do NOT call _bump() or touch st.session_state here.
            # Worker threads have no Streamlit ScriptRunContext — any call to
            # st.session_state from a background thread raises a warning and
            # silently does nothing. Counts are updated in the main thread below.
            idx, text = idx_text
            raw = self._call_api(text)
            if raw is None:
                return idx, None
            vec = self._normalise(np.array(raw, dtype=np.float32))
            key = hashlib.md5(text.encode()).hexdigest()
            self._store(key, vec)
            return idx, vec

        misses = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch, item): item[0] for item in to_fetch.items()}
            for future in as_completed(futures):
                idx, vec = future.result()
                results[idx] = vec
                misses += 1        # every fetch from Ollama is a cache miss
                completed += 1
                if progress_cb:
                    progress_cb(completed, total)

        # Update cache stats from the main thread where st.session_state is safe
        for _ in range(misses):
            _bump(hit=False)

        # Flush disk cache once after all parallel writes
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
            self._mem[key]  = vec
            self._disk[key] = vec

    def _call_api(self, text: str) -> Optional[List[float]]:
        """Single blocking HTTP call to Ollama. Safe to call from many threads."""
        try:
            r = requests.post(
                f"{AppConfig.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": AppConfig.EMBEDDING_MODEL, "prompt": text},
                timeout=AppConfig.EMBEDDING_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json().get("embedding")
            logger.warning("Embeddings API returned HTTP %s.", r.status_code)
        except requests.ConnectionError:
            logger.error("Cannot reach Ollama. Is 'ollama serve' running?")
        except requests.Timeout:
            logger.error("Embedding request timed out (text length: %d).", len(text))
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
                except:
                    pass
            return {}

    def _save_disk(self) -> None:
        """Thread-safe disk flush. Called once after a batch, not per-embedding."""
        with self._disk_lock:
            try:
                with open(_EMBEDDING_CACHE, "wb") as fh:
                    pickle.dump(self._disk, fh)
            except Exception as exc:
                logger.warning("Disk cache save failed: %s", exc)


# =============================================================================
# DOCUMENT INDEX  (per-source numpy matrix store)
# =============================================================================


class DocumentIndex:
    """
    Stores all chunks for ONE source as a (N, D) numpy matrix.

    Search is a single matrix multiply: scores = matrix @ query_vec
    This is 50–100× faster than a Python loop for large sources.
    Embeddings are unit-normalised so the dot product equals cosine similarity.
    """

    def __init__(
        self,
        source_key:  str,
        source_name: str,
        category:    str,
        emb:         EmbeddingService,
        registry:    SourceRegistry,
    ) -> None:
        self._key     = source_key
        self._name    = source_name
        self._cat     = category
        self._emb     = emb
        self._registry = registry
        self._chunks: List[Chunk]          = []
        self._matrix: Optional[np.ndarray] = None   # shape (N, D)

    # ── Readiness ─────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._matrix is not None and bool(self._chunks)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> bool:
        """
        Build index from the pre-processed DataFrame produced by preprocess.py.

        Each row becomes one Chunk. The text sent to the embedding model is:
          "<section> (<part>): <section_title>\\n\\n<content>"
        This matches the format seen during queries so retrieval is accurate.
        """
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

            # Build the text that will be embedded
            header = f"{section}"
            if part:
                header += f" ({part})"
            if title:
                header += f": {title}"
            chunk_text = f"{header}\n\n{content}" if header.strip(": ") else content

            raw.append(Chunk(
                text        = chunk_text,
                source_key  = self._key,
                source_name = self._name,
                category    = self._cat,
                section     = section,
                part        = part,
                relates_to_acts  = getattr(self._registry.get(self._key), 'relates_to_acts', []),
            ))

        return self._embed_and_build(raw)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray) -> List[SearchResult]:
        if self._matrix is None or not self._chunks:
            return []

        # Vectorised dot product (cosine similarity because vectors are unit-norm)
        scores: np.ndarray = self._matrix @ query_vec            # shape (N,)

        # Efficient partial sort: get the top candidates without full sort
        k      = min(AppConfig.TOP_K_PER_SOURCE * 3, len(scores))
        top    = np.argpartition(scores, -k)[-k:]
        top    = top[np.argsort(scores[top])[::-1]]

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
        try:
            with open(self._cache_path, "wb") as fh:
                pickle.dump({"chunks": self._chunks, "matrix": self._matrix}, fh)
            logger.info("Saved index '%s'.", self._key)
        except Exception as exc:
            logger.warning("Could not save index '%s': %s", self._key, exc)

    def load(self) -> bool:
        if not os.path.exists(self._cache_path):
            return False
        try:
            with open(self._cache_path, "rb") as fh:
                payload = pickle.load(fh)
            self._chunks = payload["chunks"]
            self._matrix = payload["matrix"]
            logger.info(
                "Loaded index '%s': %d chunks.", self._key, len(self._chunks)
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load index '%s': %s", self._key, exc)
            return False

    def delete_cache(self) -> None:
        if os.path.exists(self._cache_path):
            os.remove(self._cache_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _embed_and_build(self, raw: List[Chunk]) -> bool:
        """
        Embed all chunks in parallel then build the numpy matrix.

        Progress is logged every 10% so you can see it moving in the console.
        The disk cache is flushed once at the end of embed_batch (not per chunk),
        which avoids hundreds of slow pickle writes.
        """
        total = len(raw)
        logger.info("  Embedding %d chunks (workers=%d) …", total, AppConfig.EMBEDDING_WORKERS)

        last_pct = [0]   # mutable cell for the closure below

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
    Subsequent queries use the in-memory numpy matrix — instant.
    """

    def __init__(self, registry: SourceRegistry, emb: EmbeddingService) -> None:
        self._registry  = registry
        self._emb       = emb
        self._indexes:  Dict[str, DocumentIndex] = {}

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def get_or_build(self, source: SourceEntry) -> Optional[DocumentIndex]:
        """Return a ready index, building it if needed."""
        # Already in memory
        if source.key in self._indexes and self._indexes[source.key].is_ready():
            return self._indexes[source.key]

        idx = DocumentIndex(source.key, source.name, source.category, self._emb, self._registry)

        # Try loading from disk cache first (fast)
        if idx.load():
            self._indexes[source.key] = idx
            return idx

        # Build from the pre-processed CSV (slower, done once)
        if not source.is_ready:
            logger.warning(
                "Source '%s' has no processed CSV. "
                "Run:  python preprocess.py --key %s",
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
        
        Returns: "Companies Act 2016" | "LLP Act 2012" | "Registration Act 1956" | None
        """
        query_lower = query.lower()
        
        # Keywords mapping
        act_keywords = {
            "Companies Act 2016": [
                "company", "companies", "director", "shareholder", "shares",
                "incorporation", "memorandum", "articles", "board meeting",
                "annual general meeting", "agm", "companies act", "act 777"
            ],
            "LLP Act 2012": [
                "llp", "limited liability partnership", "partnership",
                "partner", "llp act", "act 743"
            ],
            "Registration of Businesses Act 1956": [
                "sole proprietor", "sole proprietorship", "business",
                "registration of business", "act 197"
            ]
        }
        
        for act, keywords in act_keywords.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches >= 2:  # Need at least 2 keyword matches
                return act
        
        return None

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:           str,
        selected_keys:   List[str],
        cat_filter:      Optional[List[str]] = None,
    ) -> Dict:
        """
        Embed the query, search all selected indexes, merge and rank results.

        Returns:
          {
            "chunks":         List[str]          - top texts for the LLM prompt
            "results":        List[SearchResult] - full detail for UI display
            "citations":      List[str]          - source names used
            "categories_hit": List[str]          - categories that returned hits
            "found":          bool
          }
        """
        query_vec = self._emb.embed(query)
        if query_vec is None:
            return _empty()
        
        detected_act = self._detect_query_act(query)
        logger.info(f"Detected user is asking about: {detected_act or 'unspecified'}")

        hits_by_source: Dict[str, List[Tuple[SearchResult, SourceEntry]]] = {}

        for source in self._registry.all_enabled():
            if source.key not in selected_keys:
                continue
            if cat_filter and source.category not in cat_filter:
                continue

            idx = self.get_or_build(source)
            if idx is None:
                continue

            # ← NEW: Check if this source relates to the detected Act
            source_relates_to = getattr(source, 'relates_to_acts', [])
            
            if detected_act and source_relates_to:
                # User asked about specific Act, but source relates to different Acts
                if detected_act not in source_relates_to:
                    # Only include this source if:
                    # 1. It's the main Act (source.category == "Legislations")
                    # 2. OR it relates to multiple Acts (guidelines, general docs)
                    is_main_act = source.category == "Legislations"
                    relates_to_multiple = len(source_relates_to) > 1
                    
                    if not (is_main_act or relates_to_multiple):
                        logger.info(
                            f"  Skipping '{source.name}' (relates to {source_relates_to}, "
                            f"but query is about {detected_act})"
                        )
                        continue

            source_hits = []
            for result in idx.search(query_vec):
                source_hits.append((result, source))

            if source_hits:
                hits_by_source[source.key] = source_hits
                logger.info(
                    "  Source '%s': %d results",
                    source.name, len(source_hits)
                )

        all_hits: List[Tuple[SearchResult, SourceEntry]] = []

        for source_key, source_results in hits_by_source.items():
            for result, source in source_results[:AppConfig.TOP_K_PER_SOURCE]:
                all_hits.append((result, source))

        # Global rank by score
        all_hits.sort(key=lambda x: x[0].score, reverse=True)

        # Deduplicate and cap
        seen:          set            = set()
        chunks:        List[str]      = []
        results:       List[SearchResult] = []
        citations:     List[str]      = []
        cats_hit:      List[str]      = []

        for result, source in all_hits:
            if result.chunk.text in seen:
                continue
            seen.add(result.chunk.text)
            chunks.append(result.chunk.text)
            results.append(result)
            if source.name not in citations:
                citations.append(source.name)
            if source.category not in cats_hit:
                cats_hit.append(source.category)
            if len(chunks) >= AppConfig.GLOBAL_TOP_K:
                break

        return {
            "chunks":         chunks,
            "results":        results,
            "citations":      citations,
            "categories_hit": cats_hit,
            "found":          bool(chunks),
        }

    # ── Status ────────────────────────────────────────────────────────────────

    def index_status(self) -> Dict[str, str]:
        """
        Returns {source_key: status_string} for all enabled sources.
        Possible statuses: "ready", "cached", "needs_preprocess", "not_indexed"
        """
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
    return {
        "chunks": [], "results": [], "citations": [],
        "categories_hit": [], "found": False,
    }


# =============================================================================
# LLM SERVICE
# =============================================================================


class LLMService:
    """
    Builds the prompt and calls Ollama.
    The system prompt enforces strict grounding: every claim must be quoted
    verbatim from the CONTEXT block with a section citation.
    """

    _SYSTEM = """\
You are a precise legal reference assistant for Suruhanjaya Syarikat Malaysia (SSM).
Your sole purpose is to provide information that is EXPLICITLY stated in the documents
provided in the CONTEXT block below.

ABSOLUTE RULES  -  YOU MUST FOLLOW ALL OF THESE WITHOUT EXCEPTION

RULE 1 - EXACT WORDING ONLY
  • Use ONLY the exact words, phrases, and sentences from the CONTEXT.
  • Do NOT paraphrase, reword, summarise, or interpret legal text.
  • Do NOT draw on general knowledge outside the CONTEXT.

RULE 2 - MANDATORY CITATIONS
  • Every factual statement MUST be followed by its citation in bold.
  • Format:  **(Section 14(1), Companies Act 2016)**
             **(Section 7, Registration of Businesses Act 1956)**
             **(Practice Note No. 3/2018, paragraph 8)**
  • If multiple sources support a statement, cite all of them.

RULE 3 - OUT-OF-SCOPE RESPONSE
  • If the answer is NOT present in the CONTEXT, respond with EXACTLY:
    "This information is not found in the provided documents.
     Please consult a licensed professional or contact SSM directly at www.ssm.com.my."
  • Do NOT attempt to answer from memory or inference.

RULE 4 - ACT IDENTIFICATION
  Before answering, identify which Act applies:
  • Companies / Directors / Shareholders / Company Secretary
      → Companies Act 2016 (Act 777)
  • Businesses / Sole Proprietorship / Partnership / Business Registration
      → Registration of Businesses Act 1956 (Act 197)
  • LLP / Limited Liability Partnership
      → Limited Liability Partnership Act 2012 (Act 743)
  • Operational or procedural matters may additionally be governed by
    Practice Notes, Pratice Directives, Guidelines, or Circulars issued by SSM.

RULE 5 - RESPONSE FORMAT (MANDATORY — follow this structure exactly)
  Use markdown. The response will be rendered in a web interface that supports
  **bold**, bullet points, numbered lists, and headings.

  Structure every response as follows:

  ## [Applicable Act or Document Name]

  **Direct Answer**
  One or two sentences using exact wording from the Act, with citation.

  **Relevant Provisions**
  Quote or closely paraphrase the exact section(s), each on its own line:
  - Section X(Y): "[exact text from the Act]" **(Section X(Y), Act Name)**
  - Section X(Z): "[exact text]" **(Section X(Z), Act Name)**

  **Practical Notes** *(only include if directly supported by the CONTEXT)*
  - Any conditions, exceptions, or deadlines stated in the Act.

  ---
  *For professional assistance, consult a Licensed Secretary or a member of
  the Professional Bodies listed in the 4th Schedule of the Companies Act 2016.*

  *This response is for informational purposes only and does not constitute
  legal advice. Always verify against the current official legislation.*

RULE 6 - NO THINKING OUT LOUD
  Do NOT output any internal reasoning, deliberation, or chain-of-thought.
  Go directly to the formatted answer. Never output <think> tags or similar.
"""

    def generate(self, query: str, context_chunks: List[str], citations: Optional[List[str]] = None) -> str:
        if not context_chunks:
            return (
                "No relevant sections were found in the selected knowledge sources. "
                "Please rephrase your question, enable more sources, or consult a "
                "licensed professional."
            )

        # BUILD CONTEXT WITH SOURCE NAMES
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source_name = citations[i] if citations and i < len(citations) else f"Source {i+1}"
            context_parts.append(f"[SOURCE: {source_name}]\n{chunk}")
        
        context_block = "\n\n".join(context_parts)

        prompt = (
            f"{self._SYSTEM}"
            f"CONTEXT FROM SSM OFFICIAL DOCUMENTS:\n"
            f"{'─' * 72}\n"
            f"{context_block}\n"
            f"{'─' * 72}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"ANSWER (use ONLY exact wording from CONTEXT above; cite every statement):\n"
        )

        raw = self._call(prompt)
        return self._postprocess(raw)

    @staticmethod
    def _postprocess(text: str) -> str:
        """
        Clean the raw LLM output before storing and displaying it.

        Steps
        -----
        1. Strip deepseek-r1 / reasoning model chain-of-thought blocks:
           <think>…</think>, <thinking>…</thinking>, <reasoning>…</reasoning>

        2. Collapse runs of 3+ blank lines to 2.

        3. Detect truncated responses (model hit token limit mid-sentence)
           and append a notice so the user knows the answer was cut off,
           rather than silently displaying an incomplete legal statement.
        """
        # ── Remove thinking / reasoning blocks ───────────────────────────────
        text = re.sub(r"<think>.*?</think>",           "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<thinking>.*?</thinking>",     "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<reasoning>.*?</reasoning>",   "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<reflection>.*?</reflection>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # ── Collapse excessive blank lines ────────────────────────────────────
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        # ── Detect truncation ────────────────────────────────────────────────
        # A response is likely truncated when it ends without sentence-closing
        # punctuation (. ! ?) and without a markdown structural closer (--- or ```)
        # This catches mid-word and mid-sentence cutoffs.
        if text and not re.search(r"[.!?)\]`-]\s*$", text):
            text += (
                "\n\n---\n"
                "⚠️ *The response was cut off before completion. "
                "Try asking a more specific question, or break it into smaller parts.*"
            )

        return text

    def _call(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{AppConfig.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model":  AppConfig.LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": AppConfig.LLM_TEMPERATURE,
                        "top_p":       AppConfig.LLM_TOP_P,
                        "top_k":       AppConfig.LLM_TOP_K,
                        "num_predict": AppConfig.LLM_MAX_TOKENS,
                        "stop":        ["USER QUESTION:", "CONTEXT FROM"],
                    },
                },
                timeout=AppConfig.LLM_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            return f"❌ LLM returned HTTP {r.status_code}."
        except requests.Timeout:
            return "❌ The model took too long. Try a shorter or simpler question."
        except requests.ConnectionError:
            return "❌ Cannot reach Ollama. Run 'ollama serve' in your terminal."
        except Exception as exc:
            logger.error("LLM error: %s", exc, exc_info=True)
            return f"❌ Unexpected error: {exc}"

# =============================================================================
# CACHE PRE-BUILDER  (runs on app startup)
# =============================================================================

class CacheBuilder:
    """Smart cache builder - shows progress only if needed."""
    
    @staticmethod
    def ensure_indexes_ready(kb: KnowledgeBase, registry: SourceRegistry) -> bool:
        """
        Check if all indexes are ready. If not, build them with progress UI.
        Returns True when ready.
        """
        enabled_sources = registry.all_enabled()
        
        # Check if all ready
        all_ready = all(
            (idx := kb.get_or_build(s)) and idx.is_ready()
            for s in enabled_sources
        )
        
        if all_ready:
            logger.info("✅ All indexes already ready (cached from disk)")
            return True
        
        # If not ready, show progress and build
        logger.info("🔨 Building indexes (first time or cache cleared)...")
        
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### ⏳ Initializing Knowledge Base (First Time Only)")
            st.info("This takes 1-2 minutes on first run, then everything is cached.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, source in enumerate(enabled_sources):
                progress = (i + 1) / len(enabled_sources)
                
                status_text.markdown(f"**Building:** {source.name}")
                logger.info(f"Building index for {source.name}...")
                
                try:
                    idx = kb.get_or_build(source)
                    if idx and idx.is_ready():
                        status_text.markdown(f"✅ **Done:** {source.name}")
                    else:
                        status_text.markdown(f"⚠️ **Skipped:** {source.name} (no processed CSV)")
                except Exception as e:
                    logger.error(f"Error building {source.key}: {e}")
                    status_text.markdown(f"❌ **Error:** {source.name}")
                
                progress_bar.progress(progress)
            
            progress_container.empty()
            st.success("✅ Knowledge base ready! Refresh to start asking questions.")
        
        return True



# =============================================================================
# STORAGE SERVICE
# =============================================================================


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
        try:
            with open(_CHAT_HISTORY_FILE, "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2, default=str, ensure_ascii=False)
        except Exception as exc:
            logger.error("Save history failed: %s", exc)

    @staticmethod
    def log_qa(
        query:   str,
        answer:  str,
        sources: List[str],
        rating:  Optional[int] = None,
    ) -> None:
        row = {
            "timestamp":    datetime.now().isoformat(),
            "question":     query[:500],
            "answer":       answer[:1000],
            "sources_used": ", ".join(sources),
            "user_rating":  rating if rating is not None else "",
        }
        try:
            new_df = pd.DataFrame([row])
            if os.path.exists(_QA_LOG_FILE):
                existing = pd.read_csv(_QA_LOG_FILE)
                new_df = pd.concat([existing, new_df], ignore_index=True)
            new_df.to_csv(_QA_LOG_FILE, index=False, encoding="utf-8")
        except Exception as exc:
            logger.error("QA log failed: %s", exc)


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
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()

# ── Singleton services (one instance per server process) ─────────────────────


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


# =============================================================================
# CSS
# =============================================================================

_CSS = """
<style>
/* ── Variables ──────────────────────────────────────────────────── */
:root {
    --primary:   #10a37f;
    --primary-d: #0d9268;
    --text:      #0d0d0d;
    --muted:     #565869;
    --border:    #e0e0e0;
    --dark:      #1a1a1a;
    --dark2:     #242424;
    --ai-bg:     #f9fdf7;
    --badge-bg:  #e8f5f0;
    --badge-tx:  #0d6e53;
    --warn-bg:   #fff8e1;
}

/* ── Layout ─────────────────────────────────────────────────────── */
.block-container { 
    padding: 18px 36px; 
    max-width: 960px; 
    margin: 0 auto; }

/* ── Header ─────────────────────────────────────────────────────── */
.hdr {
    text-align: center;
    padding: 22px 0 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.hdr h1 { font-size: 2.1rem; font-weight: 700; color: var(--text); margin: 0; }
.hdr p  { font-size: 0.88rem; color: var(--muted); margin-top: 6px; }

/* ── Chat bubbles ────────────────────────────────────────────────── */
.mu { display: flex; justify-content: flex-end; margin-bottom: 8px; }
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
.ma { display: flex; justify-content: flex-start; margin-bottom: 8px; }
.ma .b {
    background: var(--ai-bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 14px 14px 14px 4px;
    padding: 10px 15px;
    max-width: 72%;
    font-size: 0.93rem;
    line-height: 1.6;
    word-wrap: break-word;
    white-space: pre-wrap;
}

/* ── Source / category badges ────────────────────────────────────── */
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

/* ── Status indicators ───────────────────────────────────────────── */
.dot-ready   { color: #2ecc71; }
.dot-cached  { color: #f39c12; }
.dot-missing { color: #e74c3c; }

/* ── Welcome screen ──────────────────────────────────────────────── */
.welcome {
    text-align: center;
    padding: 52px 20px;
    color: var(--muted);
}
.welcome h2 { color: #444; font-size: 1.5rem; margin-bottom: 10px; }
.welcome .eg {
    font-size: 0.87rem;
    color: #888;
    margin-top: 14px;
    font-style: italic;
    line-height: 1.8;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--dark);
    border-right: 1px solid #2e2e2e;
}
[data-testid="stSidebar"] * { color: #ccc !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b { color: #fff !important; }
[data-testid="stSidebar"] .stCheckbox label { font-size: 0.82rem !important; }
[data-testid="stSidebar"] .stExpander summary { font-size: 0.88rem !important; }

/* ── Preprocess warning ──────────────────────────────────────────── */
.preprocess-warn {
    background: var(--warn-bg);
    border: 1px solid #ffe082;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.83rem;
    color: #795548;
    margin: 6px 0;
}

@media (max-width: 768px) {
    .block-container { padding: 12px 14px; }
    .mu .b, .ma .b   { max-width: 90%; }
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

        # ── Category filter ────────────────────────────────────────────────
        st.markdown("**🗂️ Category Filter**")
        all_cats = st.checkbox("All categories", value=True, key="cat_all")
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
            "ready":           ("🟢", "Index in memory"),
            "cached":          ("🟡", "Index cached on disk"),
            "not_indexed":     ("🔵", "Processed, not yet indexed (will index on first query)"),
            "needs_preprocess":("🔴", "Run: python preprocess.py --key <key>"),
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
                        idx_status.get(src.key, "needs_preprocess"),
                        ("❓", ""),
                    )
                    tag = "PDF" if src.type == "pdf" else "CSV"

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
                        ok = kb.rebuild_one(options[pick])
                    (st.success if ok else st.error)(
                        f"{'✅ Done' if ok else '❌ Failed'}: {pick}"
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
        c1.link_button("SSM Portal", "https://www.ssm.com.my",       use_container_width=True)
        c2.link_button("MyCoID",     "https://www.mycoid.com.my",     use_container_width=True)

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
        # ── User bubble (right-aligned, green) — pure HTML ───────────────────
        st.markdown(
            f'<div class="mu"><div class="b">{msg["query"]}</div></div>',
            unsafe_allow_html=True,
        )

        # ── Assistant response — use st.chat_message so markdown renders ─────
        # We cannot use st.markdown() inside the custom <div class="ma"> because
        # Streamlit does not parse markdown inside raw HTML strings passed to
        # unsafe_allow_html. Instead we use st.chat_message which natively
        # renders **bold**, bullet lists, numbered lists, and code blocks.
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(msg["response"])

            # Category badges below the response
            cats = msg.get("categories_hit", [])
            if cats:
                badge_html = " ".join(
                    f'<span class="badge">{c}</span>' for c in cats
                )
                st.markdown(badge_html, unsafe_allow_html=True)

def _auto_scroll_to_latest() -> None:
    """Auto-scroll to latest message when new response arrives."""
    
    history = st.session_state.get("chat_history", [])
    if not history:
        return
    
    # Track message count to know when new messages arrive
    if "last_message_count" not in st.session_state:
        st.session_state["last_message_count"] = 0
    
    current_count = len(history)
    
    # Only scroll if messages were just added
    if current_count > st.session_state["last_message_count"]:
        st.session_state["last_message_count"] = current_count
        
        # Scroll to bottom with JavaScript
        scroll_js = """
        <script>
            // Scroll to bottom after a tiny delay (let content render first)
            setTimeout(function() {
                window.scrollTo(0, document.body.scrollHeight);
            }, 100);
        </script>
        """
        st.markdown(scroll_js, unsafe_allow_html=True)

def _render_feedback(latest: Dict) -> None:
    store = _store()
    with st.expander("📋 Feedback & Export", expanded=False):
        c1, c2, c3 = st.columns(3)
        if c1.button("👍 Helpful",    use_container_width=True):
            store.log_qa(latest["query"], latest["response"], latest.get("citations", []), 5)
            st.success("Thanks!")
        if c2.button("😐 Okay",       use_container_width=True):
            store.log_qa(latest["query"], latest["response"], latest.get("citations", []), 3)
            st.info("Noted.")
        if c3.button("👎 Needs work", use_container_width=True):
            store.log_qa(latest["query"], latest["response"], latest.get("citations", []), 1)
            st.warning("We'll improve.")

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

    # Ensure indexes are ready before serving queries
    kb = _kb()
    reg = _reg()
    if not CacheBuilder.ensure_indexes_ready(kb, reg):
        st.warning("Could not initialize knowledge base. Please check logs.")
        st.stop()

    # Load persisted history once per session
    if not st.session_state["chat_history"]:
        st.session_state["chat_history"] = _store().load_history()

    _render_messages()

    _auto_scroll_to_latest()

    # ── Input row ──────────────────────────────────────────────────────────
    col_txt, col_btn = st.columns([0.85, 0.15], gap="small")
    with col_txt:
        user_input: str = st.text_area(
            "Message",
            placeholder=(
                "Ask about legislation, practice notes, guidelines, circulars, FAQ, or forms …"
            ),
            height=60,
            label_visibility="collapsed",
            key="user_input",
        )
    with col_btn:
        send = st.button("Send ➤", use_container_width=True, type="primary")

    # ── Handle submission ──────────────────────────────────────────────────
    if send and user_input.strip():
        if not selected_keys:
            st.warning(
                "No active knowledge sources. "
                "Select at least one source in the sidebar, "
                "or run **python preprocess.py** to prepare unprocessed sources."
            )
            st.stop()

        kb    = _kb()
        llm   = _llm()
        store = _store()

        with st.spinner("Searching SSM documents …"):
            prog = st.progress(0, "Embedding query …")
            result = kb.search(
                query=user_input,
                selected_keys=selected_keys,
                cat_filter=cat_filter,
            )
            prog.progress(55, "Generating grounded answer …")
            response = llm.generate(
                user_input, 
                result["chunks"],
                citations=result["citations"]  # Pass source names
            )
            prog.progress(100, "Done.")

        record: Dict = {
            "query":          user_input,
            "response":       response,
            "citations":      result["citations"],
            "categories_hit": result["categories_hit"],
            "timestamp":      datetime.now().isoformat(),
        }
        st.session_state["chat_history"].append(record)
        store.save_history(st.session_state["chat_history"])
        st.rerun()

    # ── Feedback panel ─────────────────────────────────────────────────────
    if st.session_state["chat_history"]:
        _render_feedback(st.session_state["chat_history"][-1])


if __name__ == "__main__":
    main()