"""
learning_agent.py — ChatSSM Self-Improving Feedback Architecture
================================================================
Implements an online, continuous learning loop that improves the system
automatically without requiring retraining or heavy infrastructure.

ARCHITECTURE OVERVIEW
---------------------

Layer 1 — FeedbackIngestion
    Collect thumbs up/down + optional user tags.

Layer 2 — AutoDiagnosis  (runs on EVERY negative response, immediately)
    Rule-based scanner: detects form hallucination, missing forms,
    uncited answers, and truncation in <5ms without an LLM call.

Layer 3 — SelfReflection  (triggered when AutoDiagnosis finds issues)
    Small LLM (qwen3:1.7b) evaluates its own output and classifies
    the root agent that failed: retrieval | form_mapping | generation.

Layer 4 — CorrectionEngine
    Writes corrections to three stores:
    - query_boosts.json      → adjust section retrieval weights per query pattern
    - form_overrides.json    → lock/unlock form mappings for specific intents
    - prompt_patches.json    → targeted system-prompt reinforcements

Layer 5 — LearningMemory
    Persistent JSON stores (no vector DB required).  Indexed by
    md5(normalized_query) so corrections generalise to similar phrasing.

HOW IT IMPROVES OVER TIME
--------------------------
Round 1: User asks "how to register secretary" → wrong form shown → thumbs down
Round 2: AutoDiagnosis detects form_hallucination → SelfReflection blames form_mapping
Round 3: CorrectionEngine writes: {intent: "register_as_secretary", locked_form_id: "ca_form_registration_secretary"}
Round 4: IntentFormAgent._to_output_dict reads override → returns correct form
Round 5: Same query → correct form → thumbs up → positive reinforcement stored
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("chatssm.learning")

# ─── Storage paths ────────────────────────────────────────────────────────────
_DATA_DIR = "qa_data"
_QUERY_BOOSTS_FILE      = os.path.join(_DATA_DIR, "query_boosts.json")
_FORM_OVERRIDES_FILE    = os.path.join(_DATA_DIR, "form_overrides.json")
_FAILURE_MEMORY_FILE    = os.path.join(_DATA_DIR, "failure_memory.json")
_PROMPT_PATCHES_FILE    = os.path.join(_DATA_DIR, "learned_patches.json")
# Maps query_hash → correct intent action.  Written on thumbs-down; read by
# IntentFormAgent.resolve() to bypass the LLM classifier on known-wrong queries.
_QUERY_INTENT_FILE      = os.path.join(_DATA_DIR, "query_intent_corrections.json")

os.makedirs(_DATA_DIR, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
_OLLAMA_URL       = "http://localhost:11434/api/chat"
_REFLECTION_MODEL = "qwen3:1.7b"   # fast, low-resource; intent only
_REFLECTION_TIMEOUT = 20           # seconds
MIN_POSITIVES_TO_LOCK = 2          # lock a form correction after N positive confirmations
MAX_FAILURE_MEMORY    = 200        # keep only the most recent N failure records


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FailureRecord:
    """One diagnosed failure event."""
    query:          str
    query_hash:     str           # md5 of normalized query
    response_snippet: str         # first 400 chars of response
    auto_issues:    List[Dict]    # from AutoDiagnosis
    root_agent:     str           # "retrieval" | "form_mapping" | "generation" | "unknown"
    reflection_note: str          # one-sentence LLM self-reflection
    timestamp:      str
    rating:         int           # 1-5 (1 = thumbs down)
    correction_applied: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "FailureRecord":
        return FailureRecord(**{k: v for k, v in d.items() if k in FailureRecord.__dataclass_fields__})


@dataclass
class QueryBoost:
    """
    Per-query section boost.  When a query pattern systematically retrieves
    the wrong sections, we adjust DocumentIndex.search() scores inline.
    """
    query_hash:     str
    section_parent: str           # e.g. "241"
    section_child:  Optional[str] # e.g. "3" or None
    delta:          float         # positive = boost, negative = suppress
    reason:         str
    confirmed_positive: int = 0   # times this boost led to thumbs up
    confirmed_negative: int = 0


@dataclass
class FormOverride:
    """
    Locks a specific form to a specific intent action.
    Written by CorrectionEngine after diagnosing form_mapping failures.
    Read by IntentFormAgent before standard taxonomy lookup.
    """
    intent_action:  str           # matches INTENT_TAXONOMY key
    form_id:        str           # exact form_id from forms.json
    form_name:      str           # exact form name (for display)
    confidence:     float         # 0.0-1.0
    locked:         bool = False  # True = use this regardless of taxonomy score
    confirmed_positive: int = 0
    confirmed_negative: int = 0
    timestamp:      str = ""


# =============================================================================
# LAYER 2 — AUTO DIAGNOSIS  (no LLM, <5ms)
# =============================================================================

# Patterns compiled once at module load
_ACTION_RE = re.compile(
    r'\b(?:how\s+to|how\s+do\s+I|cara|register|convert|lodge|submit|'
    r'appoint|apply\s+for|change|incorporate|wind\s+up|strike\s+off)\b',
    re.IGNORECASE,
)
_LINKED_FORM_RE  = re.compile(r'\[([^\]]+)\]\(https?://[^\)]+\)', re.IGNORECASE)
_CITATION_RE     = re.compile(r'\*\*\((?:Section|Regulation|para|Q\d)', re.IGNORECASE)
_OOS_RE          = re.compile(
    r"couldn't find|not in my knowledge base|saya tidak menemui|tidak menemui maklumat",
    re.IGNORECASE,
)
_TRUNCATION_RE   = re.compile(r'cut off before completion|⚠️.*cut off', re.IGNORECASE)
_KEYWORD_INTENT_PATTERNS = [
    (re.compile(r'\bconvert\b.{0,40}\bprivate\b|\bprivate\b.{0,40}\bconvert\b', re.IGNORECASE),
     "convert_public_to_private"),
    (re.compile(r'\bconvert\b.{0,40}\bpublic\b|\bpublic\b.{0,40}\bconvert\b', re.IGNORECASE),
     "convert_private_to_public"),
    (re.compile(r'\bstrike.?off\b|\bstrike\s+off\b', re.IGNORECASE),
     "strike_off_company"),
    (re.compile(r'\bwind.?up\b|\bwinding.?up\b', re.IGNORECASE),
     "wind_up_company_voluntary"),
    (re.compile(r'\bappoint\b.{0,30}\bdirector\b', re.IGNORECASE),
     "appoint_director"),
    (re.compile(r'\bappoint\b.{0,30}\bsecretary\b', re.IGNORECASE),
     "appoint_secretary"),
    (re.compile(r'\bresign\b.{0,30}\bsecretary\b|\bsecretary\b.{0,30}\bresign\b', re.IGNORECASE),
     "secretary_resignation"),
    (re.compile(r'\bannual\s+return\b', re.IGNORECASE),
     "lodge_annual_return"),
    (re.compile(r'\bincorporat\b', re.IGNORECASE),
     "incorporate_company"),
]

def _infer_intent_from_keywords(query: str) -> str:
    """
    Keyword-based fallback for suggested_intent when self_reflect JSON fails.
    Returns the most likely intent action string, or "" if no pattern matches.
    Only used as a safety net — self_reflect is always tried first.
    """
    for pattern, intent in _KEYWORD_INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return ""

def auto_diagnose(
    query: str,
    response: str,
    forms_shown: List[Dict],
) -> List[Dict]:
    """
    Instant rule-based failure detection.  Returns a list of issue dicts.
    Each issue has: type, detail, confidence (0-1).

    Called on EVERY response that gets a thumbs-down rating.
    Also called speculatively on borderline responses (confidence check).
    """
    issues: List[Dict] = []
    form_names_shown = {f.get("name", "").lower() for f in forms_shown}

    # ── Issue 1: Form hallucination ──────────────────────────────────────────
    # A hyperlink in the response references a form NOT in the matched set.
    linked_names = {m.group(1).lower() for m in _LINKED_FORM_RE.finditer(response)}
    hallucinated = linked_names - form_names_shown
    if hallucinated:
        issues.append({
            "type": "form_hallucination",
            "detail": list(hallucinated),
            "confidence": 0.95,
            "agent": "form_mapping",
        })

    # ── Issue 2: Missing form ────────────────────────────────────────────────
    # Procedural query but no form shown and response doesn't link any.
    if (_ACTION_RE.search(query)
            and not linked_names
            and not forms_shown
            and not _OOS_RE.search(response)):
        issues.append({
            "type": "missing_form",
            "detail": "procedural query with no form shown",
            "confidence": 0.70,
            "agent": "form_mapping",
        })

    # ── Issue 3: Uncited substantive answer (hallucination risk) ─────────────
    has_content = len(response.strip()) > 120
    has_citation = bool(_CITATION_RE.search(response))
    is_oos = bool(_OOS_RE.search(response))
    if has_content and not has_citation and not is_oos:
        issues.append({
            "type": "uncited_answer",
            "detail": "substantive answer contains no source citations",
            "confidence": 0.65,
            "agent": "generation",
        })

    # ── Issue 4: Response truncation ─────────────────────────────────────────
    if _TRUNCATION_RE.search(response):
        issues.append({
            "type": "truncation",
            "detail": "response was cut off before completion",
            "confidence": 0.99,
            "agent": "generation",
        })

    return issues


# =============================================================================
# LAYER 3 — SELF-REFLECTION AGENT  (small LLM, runs async after thumbs-down)
# =============================================================================

_REFLECTION_SYSTEM = """\
You are a quality evaluator for a Malaysian legal AI assistant.
Analyze the following query + response and identify what went wrong.
Respond with ONLY a JSON object — no explanation, no markdown.

JSON schema:
{
  "root_agent": "<one of: retrieval | form_mapping | generation | unknown>",
  "issue_summary": "<one sentence: what failed and why>",
  "suggested_sections": ["<section numbers that should have been retrieved>"],
  "suggested_intent": "<taxonomy action code or empty string>"
}
"""

_INTENT_TAXONOMY_KEYS = [
    "incorporate_company", "convert_private_to_public", "convert_public_to_private",
    "convert_unlimited_to_limited", "change_company_name", "strike_off_company",
    "wind_up_company_voluntary", "wind_up_company_court", "restore_company",
    "reserve_company_name", "appoint_director", "remove_director",
    "director_resignation", "appoint_secretary", "secretary_resignation",
    "register_as_secretary", "allot_shares", "transfer_shares",
    "reduce_share_capital", "buy_back_shares", "redeem_preference_shares",
    "alter_share_capital", "lodge_annual_return", "lodge_financial_statements",
    "hold_agm", "extend_agm_deadline", "audit_exemption",
    "lodge_beneficial_ownership", "register_charge", "satisfy_charge",
    "assign_charge", "apply_judicial_management", "voluntary_arrangement",
    "change_registered_address", "register_foreign_company", "apply_extension_of_time",
    "adopt_constitution", "amend_constitution", "register_clbg", "apply_clbg_minister",
]

def _repair_and_parse_json(raw: str) -> Optional[Dict]:
    """
    4-pass JSON repair for LLM output with unescaped apostrophes,
    control characters, or bad backslashes.
    Returns None if all passes fail.
    """
    # Pass 1: direct parse (fast path for well-formed output)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Pass 2: escape literal control characters inside string values
    out: list = []
    in_str = False
    escaped = False
    _ESC_MAP = {'\n': '\\n', '\r': '\\r', '\t': '\\t'}
    for ch in raw:
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == '\\' and in_str:
            out.append(ch)
            escaped = True
        elif ch == '"':
            in_str = not in_str
            out.append(ch)
        elif in_str and ch in _ESC_MAP:
            out.append(_ESC_MAP[ch])
        elif in_str and ord(ch) < 0x20:
            pass   # drop remaining C0 control chars
        else:
            out.append(ch)
    try:
        return json.loads(''.join(out))
    except json.JSONDecodeError:
        pass

    # Pass 3: strip C0 control chars and lone backslashes
    try:
        c = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
        c = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', c)
        return json.loads(c)
    except json.JSONDecodeError:
        pass

    # Pass 4: combine passes 2 and 3
    try:
        fixed = ''.join(out)  # result from pass 2
        fixed = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', fixed)
        fixed = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None

def self_reflect(
    query: str,
    response: str,
    auto_issues: List[Dict],
    forms_shown: List[Dict],
) -> Dict:
    """
    Asks the small LLM to evaluate its own failure.
    Returns parsed reflection dict, or a safe default on any error.
    """
    form_context = ", ".join(f.get("name", "") for f in forms_shown) or "none"
    auto_summary = "; ".join(i["type"] for i in auto_issues) or "none detected automatically"

    user_msg = (
        f"Query: {query}\n\n"
        f"Response (first 600 chars): {response[:600]}\n\n"
        f"Forms shown: {form_context}\n"
        f"Auto-detected issues: {auto_summary}\n\n"
        f"Valid intent taxonomy codes: {', '.join(_INTENT_TAXONOMY_KEYS)}"
    )

    try:
        resp = requests.post(
            _OLLAMA_URL,
            json={
                "model": _REFLECTION_MODEL,
                "messages": [
                    {"role": "system", "content": _REFLECTION_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                "stream": False,
                "think":  False,
                "options": {"temperature": 0.0, "num_predict": 400, "num_ctx": 2048},
            },
            timeout=(5, _REFLECTION_TIMEOUT),
        )
        if resp.status_code != 200:
            return _default_reflection()
        raw = resp.json().get("message", {}).get("content", "").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"<think>.*$",         "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        # Extract the JSON object — ignore any preamble text the model added
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]

        data = _repair_and_parse_json(raw)
        if data is None:
            logger.warning("SelfReflection: JSON repair failed, using default.")
            return _default_reflection()

        if data.get("root_agent") not in ("retrieval", "form_mapping", "generation", "unknown"):
            data["root_agent"] = "unknown"
        return {
            "root_agent":         data.get("root_agent", "unknown"),
            "issue_summary":      str(data.get("issue_summary", "")),
            "suggested_sections": [str(s) for s in data.get("suggested_sections", [])],
            "suggested_intent":   str(data.get("suggested_intent", "")),
        }

    except Exception as exc:
        logger.warning("SelfReflection failed: %s", exc)
        return _default_reflection()


def _default_reflection() -> Dict:
    return {
        "root_agent": "unknown",
        "issue_summary": "Reflection unavailable",
        "suggested_sections": [],
        "suggested_intent": "",
    }


# =============================================================================
# LAYER 4 — CORRECTION ENGINE
# =============================================================================

class CorrectionEngine:
    """
    Applies targeted corrections based on diagnosed failures.
    All corrections are written to lightweight JSON stores — no DB needed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._boosts:    Dict[str, List[QueryBoost]] = self._load_json(_QUERY_BOOSTS_FILE)
        self._overrides: Dict[str, FormOverride]     = self._load_form_overrides()
        # query_hash → correct intent action (written on thumbs-down, read by IntentFormAgent)
        self._intent_corrections: Dict[str, str]     = self._load_intent_corrections()
        self._intent_confirmations: Dict[str, int]   = self._load_intent_corrections(
            key="_confirmations"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply_correction(
        self,
        failure: FailureRecord,
        reflection: Dict,
        forms_shown: List[Dict],
    ) -> None:
        """Entry point: routes to the right correction handler based on root_agent."""
        agent = reflection.get("root_agent", "unknown")
        suggested_intent = reflection.get("suggested_intent", "")

        if agent == "form_mapping":
            self._correct_form_mapping(failure, reflection, forms_shown, suggested_intent)
        elif agent == "retrieval":
            self._correct_retrieval(failure, reflection)
        elif agent == "generation":
            self._correct_generation(failure)

        logger.info(
            "CorrectionEngine: applied correction for '%s' (agent=%s)",
            failure.root_agent, agent,
        )

    def apply_positive_reinforcement(self, query_hash: str, forms_shown: List[Dict]) -> None:
        """
        Called on thumbs-up. Does three things:
        1. Confirms any stored query-intent correction for this query_hash.
        2. Propagates the confirmation to any FormOverride linked to the
        corrected intent, eventually locking it.
        3. Clears the query_hash from _NEGATIVELY_RATED (via the return value
        used by the caller) so the cache works normally again.
        """
        with self._lock:
            # ── 1. Confirm query-intent correction ────────────────────────────────
            correct_intent = self._intent_corrections.get(query_hash)
            if correct_intent:
                count = self._intent_confirmations.get(query_hash, 0) + 1
                self._intent_confirmations[query_hash] = count
                self._save_intent_corrections()
                logger.info(
                    "CorrectionEngine: intent correction hash=%s → '%s' "
                    "confirmed %d time(s)",
                    query_hash, correct_intent, count,
                )
            else:
                correct_intent = None

            # ── 2. Propagate to FormOverride (fix: use intent match, not hash match)
            for intent_action, override in self._overrides.items():
                # Match via the corrected intent stored for this query,
                # OR by direct intent_action name match (for future use)
                if correct_intent and intent_action == correct_intent:
                    override.confirmed_positive += 1
                    if override.confirmed_positive >= MIN_POSITIVES_TO_LOCK:
                        override.locked = True
                        logger.info(
                            "CorrectionEngine: LOCKED form override '%s' → '%s' "
                            "after %d confirmations",
                            intent_action, override.form_name,
                            override.confirmed_positive,
                        )
            self._save_overrides()

    def get_form_override(self, intent_action: str) -> Optional[FormOverride]:
        """
        Called by IntentFormAgent before standard taxonomy lookup.
        Returns a locked override if one exists for this intent.
        """
        return self._overrides.get(intent_action)

    def get_correct_intent(self, query_hash: str) -> Optional[str]:
        """
        Returns the correct intent action for a query_hash that was previously
        diagnosed as wrong, or None if no correction is stored.
        Called by IntentFormAgent.resolve() BEFORE the LLM classifier to bypass
        repeated misclassifications on known-wrong query patterns.
        """
        return self._intent_corrections.get(query_hash)

    def get_section_boost(self, query_hash: str, section_parent: str) -> float:
        """
        Returns the boost delta for a specific section in the context of a query.
        Positive = boost this section higher; negative = suppress it.
        Called by DocumentIndex.search() after standard scoring.
        """
        boosts = self._boosts.get(query_hash, [])
        for b in boosts:
            if isinstance(b, dict):
                if b.get("section_parent") == section_parent:
                    return float(b.get("delta", 0.0))
            else:
                if b.section_parent == section_parent:
                    return b.delta
        return 0.0

    # ── Internal handlers ──────────────────────────────────────────────────────

    def _correct_form_mapping(self, failure, reflection, forms_shown, suggested_intent):
        # Record negative signals for every form that was shown and rated wrong
        for f in forms_shown:
            fid = f.get("form_id") or f.get("name", "")
            if not fid:
                continue
            with self._lock:
                # Increment negative count on any existing override for this form
                existing = next(
                    (ov for ov in self._overrides.values() if ov.form_id == fid),
                    None,
                )
                if existing:
                    existing.confirmed_negative += 1
                    logger.info(
                        "CorrectionEngine: incremented negative count for form '%s' (n=%d)",
                        fid, existing.confirmed_negative,
                    )
            self._save_overrides()

        # Also record intent-level failure if intent is known
        if not suggested_intent or suggested_intent not in _INTENT_TAXONOMY_KEYS:
            return
        override_key = suggested_intent
        if override_key not in self._overrides:
            with self._lock:
                self._overrides[override_key] = FormOverride(
                    intent_action  = suggested_intent,
                    form_id        = "",          # unknown — user didn't say which is correct
                    form_name      = f"[flagged: {suggested_intent}]",
                    confidence     = 0.5,
                    locked         = False,
                    timestamp      = datetime.now().isoformat(),
                )
                self._save_overrides()
        logger.info("CorrectionEngine: recorded form_mapping failure for intent '%s'", suggested_intent)

        # ── Write query-hash → correct-intent correction ──────────────────────
        # This is the critical fix: maps the exact query pattern to the correct
        # intent so IntentFormAgent can bypass the LLM on re-runs of the same query.
        qh = failure.query_hash
        with self._lock:
            self._intent_corrections[qh] = suggested_intent
            self._save_intent_corrections()
        logger.info(
            "CorrectionEngine: stored query_hash=%s → correct_intent='%s'",
            qh, suggested_intent,
        )

        # ── Write a classifier-prompt patch to learned_patches.json ───────────
        # This patches the IntentFormAgent's LLM system prompt so similar
        # (but not identical) phrasings also get the right classification.
        self._write_classifier_patch(failure.query, suggested_intent)

    def _write_classifier_patch(self, query: str, correct_intent: str) -> None:
        """
        Appends a targeted correction note to learned_patches.json so the
        IntentFormAgent's classifier LLM sees it on every future request.
        Deduplicates by intent pair so the file doesn't grow unbounded.
        """
        patches = {}
        try:
            if os.path.exists(_PROMPT_PATCHES_FILE):
                with open(_PROMPT_PATCHES_FILE, "r", encoding="utf-8") as fh:
                    patches = json.load(fh)
        except Exception:
            pass

        patch_key = f"intent_correction_{correct_intent}"
        if patch_key not in patches:
            # Trim the query to a representative phrase — avoid leaking PII
            query_hint = re.sub(r'\s+', ' ', query.lower().strip())[:80]
            patches[patch_key] = {
                "type":           "intent_direction_correction",
                "correct_intent": correct_intent,
                "example_query":  query_hint,
                "patch_text": (
                    f"CORRECTION (from user feedback): "
                    f"Queries like \"{query_hint}\" should map to "
                    f"\"{correct_intent}\". "
                    f"Read the direction (to/from public/private) very carefully."
                ),
                "timestamp": datetime.now().isoformat(),
            }
            with self._lock:
                self._atomic_write(_PROMPT_PATCHES_FILE, patches)
            logger.info(
                "CorrectionEngine: wrote classifier patch for intent '%s'", correct_intent
            )

    @staticmethod
    def _load_intent_corrections(key: str = "_corrections") -> Dict:
        """Load query_hash → correct_intent mapping from disk."""
        try:
            if os.path.exists(_QUERY_INTENT_FILE):
                with open(_QUERY_INTENT_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if "_corrections" in data:
                    return data.get(key, {})
                return data if key == "_corrections" else {}
        except Exception as exc:
            logger.warning("CorrectionEngine: could not load intent corrections: %s", exc)
        return {}

    def _save_intent_corrections(self) -> None:
        self._atomic_write(_QUERY_INTENT_FILE, {
            "_corrections":   self._intent_corrections,
            "_confirmations": self._intent_confirmations,
        })

    def _correct_retrieval(self, failure: FailureRecord, reflection: Dict) -> None:
        """
        Suppresses sections that were over-retrieved for this query pattern.
        Uses the sections the LLM suggested should have been retrieved instead.
        """
        suggested = reflection.get("suggested_sections", [])
        if not suggested:
            return

        qh = failure.query_hash
        existing = {}
        for b in self._boosts.get(qh, []):
            section_parent = b.get("section_parent") if isinstance(b, dict) else b.section_parent
            existing[section_parent] = b

        for sec in suggested:
            sec_norm = re.sub(r'^(?:section|seksyen|s\.?)\s*', '', str(sec).lower()).strip()
            if sec_norm and sec_norm not in existing:
                boost = QueryBoost(
                    query_hash=qh, section_parent=sec_norm, section_child=None,
                    delta=+0.15, reason=f"suggested by self-reflection: {failure.query[:60]}",
                )
                self._boosts.setdefault(qh, []).append(boost)

        with self._lock:
            self._save_boosts()

    def _correct_generation(self, failure: FailureRecord) -> None:
        """
        Writes a short-lived prompt patch for generation failures.
        The PromptOptimizer already handles this via failure_type counts,
        so we only need to ensure the failure_type is classified correctly.
        """
        # This is handled by FeedbackStore + PromptOptimizer in chatssm_app.py.
        # We log here for completeness.
        logger.info("CorrectionEngine: generation failure logged for PromptOptimizer.")

    # ── Persistence ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: str) -> Dict:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception as exc:
            logger.warning("CorrectionEngine load failed (%s): %s", path, exc)
        return {}

    def _load_form_overrides(self) -> Dict[str, FormOverride]:
        raw = self._load_json(_FORM_OVERRIDES_FILE)
        result = {}
        for k, v in raw.items():
            try:
                result[k] = FormOverride(**{f: v[f] for f in FormOverride.__dataclass_fields__ if f in v})
            except Exception:
                pass
        return result

    def _save_boosts(self) -> None:
        serializable = {
            qh: [asdict(b) if not isinstance(b, dict) else b for b in boosts]
            for qh, boosts in self._boosts.items()
        }
        self._atomic_write(_QUERY_BOOSTS_FILE, serializable)

    def _save_overrides(self) -> None:
        self._atomic_write(
            _FORM_OVERRIDES_FILE,
            {k: asdict(v) for k, v in self._overrides.items()}
        )

    @staticmethod
    def _atomic_write(path: str, data: Dict) -> None:
        fd, tmp = None, None
        try:
            fd, tmp = tempfile.mkstemp(dir=_DATA_DIR, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = None
                json.dump(data, fh, indent=2, ensure_ascii=False)
            shutil.move(tmp, path)
            tmp = None
        except Exception as exc:
            logger.warning("CorrectionEngine write failed (%s): %s", path, exc)
            if fd:
                try: os.close(fd)
                except OSError: pass
            if tmp and os.path.exists(tmp):
                try: os.unlink(tmp)
                except OSError: pass


# =============================================================================
# LAYER 5 — LEARNING MEMORY
# =============================================================================

class LearningMemory:
    """
    Persists failure records for trend analysis and pattern detection.
    Bounded at MAX_FAILURE_MEMORY to prevent unbounded growth.
    """

    def __init__(self) -> None:
        self._records: List[FailureRecord] = self._load()
        self._lock = threading.Lock()

    def record(self, failure: FailureRecord) -> None:
        with self._lock:
            self._records.append(failure)
            if len(self._records) > MAX_FAILURE_MEMORY:
                self._records = self._records[-MAX_FAILURE_MEMORY:]
            self._save()

    def recent_failures(self, n: int = 10) -> List[FailureRecord]:
        return self._records[-n:]

    def failure_rate_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rec in self._records:
            for issue in rec.auto_issues:
                t = issue.get("type", "unknown")
                counts[t] = counts.get(t, 0) + 1
        return counts

    def _load(self) -> List[FailureRecord]:
        try:
            if os.path.exists(_FAILURE_MEMORY_FILE):
                with open(_FAILURE_MEMORY_FILE, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                return [FailureRecord.from_dict(d) for d in raw]
        except Exception as exc:
            logger.warning("LearningMemory load failed: %s", exc)
        return []

    def _save(self) -> None:
        try:
            fd, tmp = tempfile.mkstemp(dir=_DATA_DIR, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump([r.to_dict() for r in self._records], fh, indent=2, ensure_ascii=False)
            shutil.move(tmp, _FAILURE_MEMORY_FILE)
        except Exception as exc:
            logger.warning("LearningMemory save failed: %s", exc)


# =============================================================================
# ORCHESTRATOR — LearningAgent (public interface used by chatssm_app.py)
# =============================================================================

class LearningAgent:
    """
    Single entry point for all learning-related operations.

    Auto-learning loop (no user input required):
      👎 received  → on_negative_feedback() fires background thread
      Background   → auto_diagnose() → self_reflect() → _map_to_failure_type()
                  → _write_back_failure_type() → FeedbackStore updated
      Next query   → PromptOptimizer.get_patches() reads updated failure_type
                  → correct prompt patch injected automatically

    👍 received  → on_positive_feedback() → CorrectionEngine locks correct forms
    """

    def __init__(self, feedback_store_path: str = os.path.join("qa_data", "feedback.json")) -> None:
        self.correction           = CorrectionEngine()
        self.memory               = LearningMemory()
        self._feedback_store_path = feedback_store_path
        self._executor            = None  # lazy thread pool

    # ── Main hooks ─────────────────────────────────────────────────────────────

    def observe(
        self,
        query: str,
        response: str,
        forms_shown: List[Dict],
        rating: Optional[int] = None,
    ) -> List[Dict]:
        """
        Called after every response.  Runs auto-diagnosis immediately.
        If issues are found AND rating is None (no explicit feedback yet),
        logs speculatively for later confirmation.
        Returns auto_issues list (may be empty).
        """
        issues = auto_diagnose(query, response, forms_shown)
        return issues

    def on_negative_feedback(
        self,
        qa_id:      str,
        query:      str,
        response:   str,
        forms_shown: List[Dict],
    ) -> None:
        """
        Called immediately when user clicks 👎.
        Runs auto_diagnose + self_reflect asynchronously (non-blocking).
        """
        threading.Thread(
            target=self._process_negative,
            args=(qa_id, query, response, forms_shown),
            daemon=True,
        ).start()

    def on_positive_feedback(self, query: str, forms_shown: List[Dict]) -> None:
        """Called immediately when user clicks 👍."""
        qh = _normalize_hash(query)
        self.correction.apply_positive_reinforcement(qh, forms_shown)

    # ── Retrieval boosting API ──────────────────────────────────────────────────

    def get_section_boost(self, query_hash: str, section_parent: str) -> float:
        """
        Returns a score delta to add to DocumentIndex cosine similarity.
        Positive = surface this section higher; negative = suppress it.
        Keeps DocumentIndex.search() unchanged — boost is applied on top.
        """
        return self.correction.get_section_boost(query_hash, section_parent)

    # ── Intent correction API ──────────────────────────────────────────────────

    def get_correct_intent(self, query_hash: str) -> Optional[str]:
        """
        Returns the correct intent action for a previously misclassified query,
        or None if no correction has been stored.
        Called by IntentFormAgent.resolve() before the LLM classifier so that
        queries which have been thumbs-downed bypass the classifier entirely.
        """
        return self.correction.get_correct_intent(query_hash)

    # ── Form override API ──────────────────────────────────────────────────────

    def get_form_override(self, intent_action: str) -> Optional[FormOverride]:
        """
        Returns a FormOverride if one has been confirmed for this intent.
        IntentFormAgent should call this BEFORE standard taxonomy lookup.
        A locked override takes precedence over all other signals.
        """
        override = self.correction.get_form_override(intent_action)
        if override and override.locked:
            return override
        return None

    # ── Analytics ──────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Returns learning system stats for the admin sidebar panel."""
        return {
            "failure_counts":       self.memory.failure_rate_by_type(),
            "total_failures_logged": len(self.memory._records),
            "recent_failures":      [
                {
                    "query":       r.query[:80],
                    "root_agent":  r.root_agent,
                    "issues":      [i["type"] for i in r.auto_issues],
                    "timestamp":   r.timestamp,
                }
                for r in self.memory.recent_failures(5)
            ],
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _process_negative(
        self,
        qa_id:       str,
        query:       str,
        response:    str,
        forms_shown: List[Dict],
    ) -> None:
        """
        Runs in a background thread — does not block the UI.

        Pipeline:
          1. auto_diagnose()  — rule-based, instant (<5ms)
          2. self_reflect()   — LLM evaluates the failure (~5-15s)
          3. Map to FeedbackStore failure_type key
          4. Write failure_type back to FeedbackStore (so PromptOptimizer uses it)
          5. Apply structural correction via CorrectionEngine
        """
        try:
            auto_issues = auto_diagnose(query, response, forms_shown)
            # Fast-path: high-confidence auto diagnosis bypasses LLM reflection
            auto_root_agent = None
            best = max(auto_issues, key=lambda i: i.get("confidence", 0), default=None)
            if best and best.get("confidence", 0) >= 0.90:
                auto_root_agent = best.get("agent", None)   # "form_mapping" or "generation"

            if auto_root_agent and best:
                reflection = {
                    "root_agent": auto_root_agent,
                    "issue_summary": f"Auto-diagnosed: {best['type']}",
                    "suggested_sections": [],
                    "suggested_intent": "",
                }

                if auto_root_agent == "form_mapping":
                    full = self_reflect(query, response, auto_issues, forms_shown)
                    reflection["suggested_intent"]   = full.get("suggested_intent", "")
                    reflection["suggested_sections"] = full.get("suggested_sections", [])

            else:
                reflection = self_reflect(query, response, auto_issues, forms_shown)

            # ── Map diagnosis to FeedbackStore.FAILURE_TYPES key ──────────────
            failure_type = self._map_to_failure_type(auto_issues, reflection)

            failure = FailureRecord(
                query            = query,
                query_hash       = _normalize_hash(query),
                response_snippet = response[:400],
                auto_issues      = auto_issues,
                root_agent       = reflection.get("root_agent", "unknown"),
                reflection_note  = reflection.get("issue_summary", ""),
                timestamp        = datetime.now().isoformat(),
                rating           = 1,
            )
            self.memory.record(failure)

            if not reflection.get("suggested_intent"):
                inferred = _infer_intent_from_keywords(query)
                if inferred:
                    reflection["suggested_intent"] = inferred
                    reflection["root_agent"] = "form_mapping"
                    logger.info(
                        "LearningAgent: keyword fallback inferred intent '%s' for query: %s",
                        inferred, query[:80],
                    )
            self.correction.apply_correction(failure, reflection, forms_shown)

            # ── Write diagnosed failure_type back to FeedbackStore ────────────
            # This is what closes the loop: PromptOptimizer reads failure_type
            # from FeedbackStore. Without this write-back, negative feedback
            # records have failure_type="" and PromptOptimizer patches never fire.
            if failure_type and self._feedback_store_path:
                self._write_back_failure_type(qa_id, failure_type)

            logger.info(
                "LearningAgent: diagnosed qa_id=%s root_agent=%s failure_type=%s issues=%s",
                qa_id[:12],
                failure.root_agent,
                failure_type or "unknown",
                [i["type"] for i in auto_issues],
            )
        except Exception as exc:
            logger.error("LearningAgent._process_negative failed: %s", exc, exc_info=True)

    @staticmethod
    def _map_to_failure_type(auto_issues: List[Dict], reflection: Dict) -> str:
        """
        Map auto_diagnose issue types and self_reflect root_agent
        to the FeedbackStore.FAILURE_TYPES keys used by PromptOptimizer.

        Priority: auto_diagnose (deterministic, high confidence) first,
        then fall back to LLM reflection classification.

        FeedbackStore.FAILURE_TYPES keys:
          wrong_source | hallucination | incorrect | incomplete |
          out_of_scope | wrong_form    | other
        """
        # Map auto_diagnose issue types → FeedbackStore keys
        _ISSUE_MAP = {
            "form_hallucination": "wrong_form",
            "missing_form":       "wrong_form",
            "uncited_answer":     "hallucination",
            "wrong_facts":        "incorrect",
            "truncation":         "incomplete",
        }

        # Use highest-confidence auto_diagnose issue first
        best_issue = max(auto_issues, key=lambda i: i.get("confidence", 0), default=None)
        if best_issue and best_issue.get("confidence", 0) >= 0.65:
            mapped = _ISSUE_MAP.get(best_issue["type"])
            if mapped:
                return mapped

        # Fall back to LLM reflection root_agent
        _AGENT_MAP = {
            "form_mapping": "wrong_form",
            "retrieval":    "wrong_source",
            "wrong_facts":  "incorrect",
            "generation":   "hallucination",
        }
        root_agent = reflection.get("root_agent", "unknown")
        return _AGENT_MAP.get(root_agent, "other")

    def _write_back_failure_type(self, qa_id: str, failure_type: str) -> None:
        """
        Update the FeedbackStore record for qa_id with the auto-diagnosed failure_type.
        Uses the same FileLock + atomic write pattern as FeedbackStore itself.
        """
        import shutil as _shutil
        from filelock import FileLock as _FileLock

        feedback_file = self._feedback_store_path
        lock_file     = feedback_file + ".lock"

        if not os.path.exists(feedback_file):
            return

        try:
            with _FileLock(lock_file):
                with open(feedback_file, "r", encoding="utf-8") as fh:
                    records = json.load(fh)

                updated = False
                for record in records:
                    if record.get("qa_id") == qa_id and not record.get("failure_type"):
                        record["failure_type"]   = failure_type
                        record["auto_diagnosed"] = True
                        updated = True
                        break

                if not updated:
                    return

                tmp = feedback_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(records, fh, indent=2, ensure_ascii=False)
                _shutil.move(tmp, feedback_file)
                logger.info(
                    "LearningAgent: wrote back failure_type='%s' for qa_id=%s",
                    failure_type, qa_id[:12],
                )
        except Exception as exc:
            logger.warning("LearningAgent._write_back_failure_type failed: %s", exc)


# =============================================================================
# UTILITIES
# =============================================================================

def _normalize_hash(text: str) -> str:
    """Normalize a query string and return its 8-char md5 prefix."""
    normalized = re.sub(r"\s+", " ", text.lower().strip()).rstrip("?.! ")
    return hashlib.md5(normalized.encode()).hexdigest()[:8]