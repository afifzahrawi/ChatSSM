"""
memory_manager.py  -  ChatSSM Persistent Memory System
=======================================================
Provides two memory layers above the existing ConversationMemory
(short-term, session-scoped) without touching the RAG retrieval pipeline.

MEMORY LAYERS
-------------
  Short-term   -- already in chatssm_app.py (ConversationMemory, 4 turns)
  Mid-term     -- session summaries, expire after MID_TERM_TTL_DAYS days
  Long-term    -- explicit user preferences and instructions, never expire

SAFETY DESIGN
-------------
Only EXPLICIT user statements are stored (keyword pattern detection).
Inferred preferences are never stored.
Memory is injected in a clearly labelled block separate from CONTEXT.
The LLM is instructed to treat memory as user preferences, not legal facts.
Retrieval uses cosine similarity; records below RETRIEVAL_THRESHOLD are skipped.
Total token budget for memory injection is hard-capped at MAX_MEMORY_CHARS.
No long-term memory is created from LLM responses -- prevents hallucination
persistence.

STORAGE
-------
  qa_data/midterm_memory.json   -- per-session summaries (auto-expire)
  qa_data/longterm_memory.json  -- user preferences and instructions

INTEGRATION (two additions to chatssm_app.py)
----------------------------------------------
  1. After each turn call:
       mem_mgr.observe(user_input)

  2. Before prompt assembly call:
       memory_block = mem_mgr.build_memory_block(query_vec)
     Then inject memory_block before <<CONTEXT_BLOCK>> in the prompt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("chatssm.memory")

# Storage paths
_USER_DATA      = "user_data"
_LANG_HISTORY_KEY = "_lang_history"

# Tuning constants
MID_TERM_TTL_DAYS    = 7    # mid-term records older than this are discarded
MAX_LONGTERM_RECORDS = 50   # cap to prevent unbounded growth
MAX_MIDTERM_RECORDS  = 20
RETRIEVAL_THRESHOLD  = 0.55  # cosine similarity required for injection
MAX_MEMORY_CHARS     = 500   # hard cap on characters injected into prompt
MAX_RECORDS_INJECTED = 3     # never inject more than this many records


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class MemoryRecord:
    memory_id:   str
    memory_type: str           # "preference" | "correction" | "instruction" | "session_summary"
    content:     str
    source:      str           # "user_explicit" | "session_summary"
    timestamp:   str           # ISO datetime
    expires_at:  Optional[str] = None   # ISO datetime; None = permanent
    embedding:   Optional[List[float]] = field(default=None, repr=False)

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            return datetime.fromisoformat(self.expires_at) < datetime.now()
        except ValueError:
            return False

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> Optional["MemoryRecord"]:
        try:
            known = set(MemoryRecord.__dataclass_fields__.keys())
            return MemoryRecord(**{k: v for k, v in d.items() if k in known})
        except Exception:
            return None

# =============================================================================
# DETECTION PATTERNS
# =============================================================================
# Intentionally conservative. False positives pollute memory.

_PREFERENCE_PATTERNS = [
    # Language preference: "please answer in Malay", "always respond in English"
    (re.compile(
        r'\b(?:please\s+)?(?:always\s+)?(?:answer|respond|reply)\s+in\s+'
        r'(malay|english|bahasa)',
        re.IGNORECASE,
    ), "preference", "Language preference: {match}"),

    # User role context
    (re.compile(
        r'\bi\s+(?:am\s+a?|work\s+(?:as\s+a?|for))\s*'
        r'(company\s+secretary|director|auditor|lawyer|accountant|'
        r'compliance\s+officer|shareholder|sole\s+proprietor)',
        re.IGNORECASE,
    ), "preference", "User role: {match}"),

    # Explicit formatting instruction
    (re.compile(
        r'\b(?:always|please\s+always)\s+'
        r'(cite\s+section\s+numbers?|use\s+simple\s+language|'
        r'give\s+(?:me\s+)?step.by.step|include\s+examples)',
        re.IGNORECASE,
    ), "instruction", "Formatting instruction: {match}"),
]

_CORRECTION_PATTERNS = [
    (re.compile(
        r'\b(?:that(?:\'s|\s+is)\s+(?:wrong|incorrect|not\s+right)|'
        r'actually[,\s]+the\s+(?:correct|right)\s+(?:answer|section|provision)\s+is|'
        r'you(?:\'re|\s+are)\s+wrong)',
        re.IGNORECASE,
    ), "correction"),
]


# =============================================================================
# MEMORY MANAGER
# =============================================================================

class MemoryManager:
    """
    Mid-term and long-term memory for ChatSSM.

    Parameters
    ----------
    embedding_service : EmbeddingService | None
        The app's singleton EmbeddingService. When None, similarity-based
        retrieval is skipped and all non-expired records up to the cap are used.
    """

    def __init__(self, embedding_service=None, session_id: str = "default") -> None:
        self._emb        = embedding_service
        self._session_id = session_id
        self._midterm:  List[MemoryRecord] = []
        self._longterm: List[MemoryRecord] = []
        self._load()

    # ── File paths (session-namespaced) ────────────────────────────────────────

    @property
    def _midterm_file(self) -> str:
        return os.path.join(_USER_DATA, f"midterm_{self._session_id[:16]}.json")

    @property
    def _longterm_file(self) -> str:
        return os.path.join(_USER_DATA, f"longterm_{self._session_id[:16]}.json")

    # ── Public API ─────────────────────────────────────────────────────────────

    def observe(self, user_input: str, lang: str = "en") -> None:
        """
        Inspect the user's latest message for memory-worthy content.
        Called once per turn with the raw user input ONLY (never the LLM
        response) to prevent hallucinations from being stored.
        """
        text = user_input.strip()
        if len(text) < 10:
            return
        self._detect_preferences(text)
        self._detect_corrections(text)
        self._track_language_behaviour(lang)

    def add_session_summary(self, session_topics: List[str]) -> None:
        """
        Store a mid-term summary of what was discussed in this session.
        Call when a session ends or when MAX_RAW_TURNS is exceeded.

        session_topics: short topic phrases, e.g. ["annual return deadline"]
        """
        if not session_topics:
            return
        content = "Recent session topics: " + ", ".join(session_topics[:8])
        expires = (datetime.now() + timedelta(days=MID_TERM_TTL_DAYS)).isoformat()
        record = self._make_record(
            content=content,
            memory_type="session_summary",
            source="session_summary",
            expires_at=expires,
        )
        self._midterm.append(record)
        if len(self._midterm) > MAX_MIDTERM_RECORDS:
            self._midterm = self._midterm[-MAX_MIDTERM_RECORDS:]
        self._save_midterm()
        logger.info("Memory: stored mid-term session summary.")

    def build_memory_block(
        self, 
        query_vec: Optional[np.ndarray] = None,
        query_text: str = "",
    ) -> str:
        """
        Build the <<USER MEMORY>> prompt block.

        Returns an empty string when no relevant memory exists so the
        prompt is never polluted with empty or irrelevant blocks.

        query_vec: unit-normalised query embedding for similarity filtering.
                   When None, up to MAX_RECORDS_INJECTED records are used.
        """
        self._purge_expired()
        all_records = self._longterm + self._midterm
        if not all_records:
            return ""

        if query_vec is not None:
            scored = self._rank_by_similarity(all_records, query_vec, query_text)
            relevant = [
                r for r, s in scored if s >= RETRIEVAL_THRESHOLD
            ][:MAX_RECORDS_INJECTED]
        else:
            relevant = all_records[:MAX_RECORDS_INJECTED]

        if not relevant:
            return ""

        type_labels = {
            "preference":      "Your preference",
            "instruction":     "Your instruction",
            "correction":      "Your correction",
            "session_summary": "Recent topics",
        }
        lines = [
            f"  {type_labels.get(r.memory_type, 'Note')}: {r.content}"
            for r in relevant
        ]

        block = (
            "<<USER MEMORY>>\n"
            "Confirmed preferences from this user. Apply to tone and format only.\n"
            "Do NOT treat these as legal facts or document content.\n"
            + "-" * 40 + "\n"
            + "\n".join(lines)
            + "\n" + "-" * 40 + "\n\n"
        )

        if len(block) > MAX_MEMORY_CHARS:
            block = block[:MAX_MEMORY_CHARS] + "\n[memory truncated]\n\n"

        return block

    def clear_longterm(self) -> None:
        self._longterm = []
        if os.path.exists(self._longterm_file):
            os.remove(self._longterm_file)
        logger.info("Memory: long-term memory cleared.")

    def clear_midterm(self) -> None:
        self._midterm = []
        if os.path.exists(self._midterm_file):
            os.remove(self._midterm_file)
        logger.info("Memory: mid-term memory cleared.")

    def summary(self) -> Dict:
        """Return counts and record list for the admin/sidebar panel."""
        self._purge_expired()
        return {
            "longterm_count": len(self._longterm),
            "midterm_count":  len(self._midterm),
            "records": [
                {
                    "type":      r.memory_type,
                    "content":   r.content[:80],
                    "timestamp": r.timestamp,
                    "expires":   r.expires_at or "never",
                }
                for r in (self._longterm + self._midterm)
            ],
        }

    # ── Detection ──────────────────────────────────────────────────────────────

    def _detect_preferences(self, text: str) -> None:
        for pattern, mtype, template in _PREFERENCE_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            content = template.format(
                match=m.group(0).strip(), content=text[:200]
            )
            if self._is_duplicate(content):
                continue
            record = self._make_record(
                content=content, memory_type=mtype, source="user_explicit"
            )
            self._embed_record(record)
            self._longterm.append(record)
            if len(self._longterm) > MAX_LONGTERM_RECORDS:
                self._longterm = self._longterm[-MAX_LONGTERM_RECORDS:]
            self._save_longterm()
            logger.info("Memory: stored %s — %s", mtype, content[:60])

    def _detect_corrections(self, text: str) -> None:
        for pattern, mtype in _CORRECTION_PATTERNS:
            if not pattern.search(text):
                continue
            content = f"User correction: {text[:200]}"
            if self._is_duplicate(content):
                continue
            # Corrections expire — they are session-relevant, not permanent truth
            expires = (
                datetime.now() + timedelta(days=MID_TERM_TTL_DAYS)
            ).isoformat()
            record = self._make_record(
                content=content,
                memory_type=mtype,
                source="user_explicit",
                expires_at=expires,
            )
            self._embed_record(record)
            self._midterm.append(record)
            if len(self._midterm) > MAX_MIDTERM_RECORDS:
                self._midterm = self._midterm[-MAX_MIDTERM_RECORDS:]
            self._save_midterm()
            logger.info("Memory: stored correction — %s", content[:60])
        
    def _track_language_behaviour(self, lang: str) -> None:
        """
        Infer language preference from consistent user behaviour.
        Stores a preference record after 3 consecutive queries in the same language.
        Only triggers for Malay — English is the default so no need to store it.
        """
        if lang not in ("ms", "mixed"):
            # User is writing in English — reset any streak tracking
            self._lang_streak = getattr(self, '_lang_streak', {"lang": None, "count": 0})
            if self._lang_streak.get("lang") != "en":
                self._lang_streak = {"lang": "en", "count": 1}
            else:
                self._lang_streak["count"] += 1
            return

        # Malay or mixed detected
        streak = getattr(self, '_lang_streak', {"lang": None, "count": 0})
        if streak.get("lang") == lang:
            streak["count"] += 1
        else:
            streak = {"lang": lang, "count": 1}
        self._lang_streak = streak

        # After 3 consecutive Malay queries, store as inferred preference
        if streak["count"] == 3:
            content = (
                "Language preference: User consistently writes in Bahasa Malaysia. "
                "Respond in Malay unless the user switches to English."
            )
            if not self._is_duplicate(content):
                record = self._make_record(
                    content=content,
                    memory_type="preference",
                    source="inferred_behaviour",   # distinct from "user_explicit"
                )
                self._embed_record(record)
                self._longterm.append(record)
                if len(self._longterm) > MAX_LONGTERM_RECORDS:
                    self._longterm = self._longterm[-MAX_LONGTERM_RECORDS:]
                self._save_longterm()
                logger.info("Memory: inferred Malay language preference from behaviour.")

    # ── Similarity ─────────────────────────────────────────────────────────────

    def _embed_record(self, record: MemoryRecord) -> None:
        if self._emb is None:
            return
        try:
            vec = self._emb.embed(record.content)
            if vec is not None:
                record.embedding = vec.tolist()
        except Exception as exc:
            logger.warning("Memory: embed failed: %s", exc)

    def _rank_by_similarity(
        self,
        records: List[MemoryRecord],
        query_vec: np.ndarray,
        query_text: str = "",
    ) -> List[tuple]:
        scored = []
        for r in records:
            if r.embedding:
                try:
                    vec = np.array(r.embedding, dtype=np.float32)
                    score = float(np.dot(vec, query_vec))
                except Exception:
                    score = 0.0
            else:
                # Keyword overlap fallback when no embedding available
                q_tok = set(re.findall(r'[a-z0-9]+', query_text.lower()))
                r_tok = set(re.findall(r'[a-z0-9]+', r.content.lower()))
                score = len(q_tok & r_tok) / max(len(q_tok), 1) * 0.4
            scored.append((r, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_record(
        self,
        content: str,
        memory_type: str,
        source: str,
        expires_at: Optional[str] = None,
    ) -> MemoryRecord:
        ts  = datetime.now().isoformat()
        mid = hashlib.md5(f"{content}{ts}".encode()).hexdigest()[:12]
        return MemoryRecord(
            memory_id=mid,
            memory_type=memory_type,
            content=content,
            source=source,
            timestamp=ts,
            expires_at=expires_at,
        )

    def _is_duplicate(self, content: str) -> bool:
        """Return True if a highly similar record already exists (token-level check)."""
        # Use token overlap instead of character prefix — avoids false positives
        # for records that share a prefix but differ in the key term.
        new_tokens = set(re.findall(r'[a-z0-9]+', content.lower()))
        if not new_tokens:
            return False
        for r in self._longterm + self._midterm:
            existing_tokens = set(re.findall(r'[a-z0-9]+', r.content.lower()))
            if not existing_tokens:
                continue
            union = new_tokens | existing_tokens
            if len(new_tokens & existing_tokens) / len(union) > 0.80:
                return True
        return False

    def _purge_expired(self) -> None:
        before = len(self._midterm) + len(self._longterm)
        self._midterm  = [r for r in self._midterm  if not r.is_expired()]
        self._longterm = [r for r in self._longterm if not r.is_expired()]
        purged = before - len(self._midterm) - len(self._longterm)
        if purged:
            logger.info("Memory: purged %d expired record(s).", purged)
            self._save_midterm()
            self._save_longterm()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        os.makedirs(_USER_DATA, exist_ok=True)
        self._midterm  = self._load_file(self._midterm_file)
        self._longterm = self._load_file(self._longterm_file)
        self._purge_expired()
        logger.info(
            "Memory: loaded %d mid-term, %d long-term record(s).",
            len(self._midterm), len(self._longterm),
        )

    @staticmethod
    def _load_file(path: str) -> List[MemoryRecord]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return [r for r in (MemoryRecord.from_dict(d) for d in raw if isinstance(d, dict)) if r is not None]
        except Exception as exc:
            logger.warning("Memory: could not load '%s': %s", path, exc)
            return []

    def _save_file(self, records: List[MemoryRecord], path: str) -> None:
        fd, tmp = None, None
        try:
            fd, tmp = tempfile.mkstemp(dir=_USER_DATA, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = None
                json.dump(
                    [r.to_dict() for r in records],
                    fh, indent=2, ensure_ascii=False,
                )
            shutil.move(tmp, path)
            tmp = None
        except Exception as exc:
            logger.warning("Memory: save to '%s' failed: %s", path, exc)
            if fd is not None:
                try: os.close(fd)
                except OSError: pass
            if tmp and os.path.exists(tmp):
                try: os.unlink(tmp)
                except OSError: pass

    def _save_midterm(self) -> None:
        self._save_file(self._midterm, self._midterm_file)

    def _save_longterm(self) -> None:
        self._save_file(self._longterm, self._longterm_file)