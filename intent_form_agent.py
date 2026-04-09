# intent_form_agent.py

from __future__ import annotations

import json
import logging
import re
import time
import os
import hashlib
import learning_agent
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import requests

logger = logging.getLogger("chatssm.intent_agent")

_FORM_OVERRIDES_FILE      = os.path.join("qa_data", "form_overrides.json")
# Written by CorrectionEngine after thumbs-down; bypasses LLM on re-runs
_QUERY_INTENT_FILE        = os.path.join("qa_data", "query_intent_corrections.json")
# Written by CorrectionEngine; injected into classifier system prompt
_CLASSIFIER_PATCHES_FILE  = os.path.join("qa_data", "learned_patches.json")

_INTENT_CACHE: Dict[str, "FormResolutionResult"] = {}
_INTENT_CACHE_MAX = 200
_NEGATIVELY_RATED: set = set() 

# ─── Structured output schema ─────────────────────────────────────────────────

@dataclass
class LegalIntent:
    """
    The agent's structured understanding of what the user wants to do legally.
    This is the ONLY output the agent produces. It never names forms or URLs.
    """
    action:          str               # e.g. "convert_private_to_public"
    legal_procedure: str               # e.g. "Conversion of company status"
    likely_sections: List[str]         # e.g. ["41(2)", "190(3)"]
    confidence:      float             # 0.0 - 1.0
    reasoning:       str               # agent's explanation (for logging/debug)
    language:        str               # "en" | "ms"
    is_actionable:   bool              # True = user wants to DO something
    ambiguous:       bool              # True = agent is not sure


@dataclass
class FormResolutionResult:
    """Final output of the full agent pipeline."""
    forms:         List[Dict]          # list of {name, url, form_number}
    intent:        Optional[LegalIntent]
    resolution_path: str               # "agent" | "section_match" | "fallback" | "none"
    confidence:    float


# ─── Intent taxonomy ──────────────────────────────────────────────────────────
# This is the controlled vocabulary the agent must output.
# Grounding the agent to a fixed taxonomy prevents hallucination drift.

INTENT_TAXONOMY = {
    # Company lifecycle
    "incorporate_company":          ["14"],
    "register_company":             ["14"],
    "convert_private_to_public":    ["41(2)", "190(3)"],
    "convert_public_to_private":    ["41(1)"],
    "convert_unlimited_to_limited": ["40"],
    "change_company_name":          ["28"],
    "strike_off_company":           ["550", "549"],
    "wind_up_company_voluntary":    ["439", "443"],
    "wind_up_company_court":        ["465", "474"],
    "restore_company":              ["535"],
    "reserve_company_name":         ["27(1)", "27(4)"],

    # Director & officer changes
    "appoint_director":             ["58", "201"],
    "remove_director":              ["206"],
    "director_resignation":         ["208", "58"],
    "appoint_secretary":            ["58", "236(2)"],
    "secretary_resignation":        ["237(2)", "58"],
    "register_as_secretary":        ["241"],

    # Share-related
    "allot_shares":                 ["75", "76"],
    "transfer_shares":              ["105"],
    "reduce_share_capital":         ["117", "119"],
    "buy_back_shares":              ["127"],
    "redeem_preference_shares":     ["72"],
    "alter_share_capital":          ["84"],

    # Annual compliance
    "lodge_annual_return":          ["68"],
    "lodge_financial_statements":   ["259"],
    "hold_agm":                     ["340"],
    "extend_agm_deadline":          ["340(4), 340"],
    "audit_exemption":              ["267A"],

    # Beneficial ownership
    "lodge_beneficial_ownership":   ["60A"],
    "update_beneficial_ownership":  ["60A"],

    # Charges
    "register_charge":              ["352"],
    "satisfy_charge":               ["360(1)"],
    "assign_charge":                ["359(1)"],

    # Corporate rescue
    "apply_judicial_management":    ["408", "406"],
    "voluntary_arrangement":        ["396", "397"],

    # Address & records
    "change_registered_address":    ["46"],
    "change_business_address":      ["PD2/2017"],

    # Foreign company
    "register_foreign_company":     ["562(1)"],
    "deregister_foreign_company":   ["578"],
    "update_foreign_company":       ["567"],

    # Extension of time
    "apply_extension_of_time":      ["609"],

    # Constitution
    "adopt_constitution":           ["32"],
    "amend_constitution":           ["36"],

    # CLBG
    "register_clbg":                ["14"],
    "apply_clbg_minister":          ["45"],
}


# ─── Agent system prompt ───────────────────────────────────────────────────────

_AGENT_SYSTEM_PROMPT = """\
You are a Malaysian corporate law intent classifier for the ChatSSM system.

Your ONLY job is to identify what legal action the user wants to perform,
then return a JSON object. You do NOT answer questions. You do NOT suggest forms.
You classify intent and return structured data.

TAXONOMY (you MUST use one of these action codes, or "unknown"):
{taxonomy}

OUTPUT FORMAT (return ONLY this JSON, no other text, no markdown):
{{
  "action": "<action_code_from_taxonomy_or_unknown>",
  "legal_procedure": "<short English description of what they want to do>",
  "likely_sections": ["41(1)", "190(3)"],
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence: why you chose this action>",
  "is_actionable": <true|false>,
  "ambiguous": <false|true>
}}

CRITICAL JSON RULES — VIOLATIONS CAUSE SYSTEM FAILURE:
1. "likely_sections" MUST be an array of QUOTED STRINGS: ["41(1)", "58"]
   NEVER write unquoted values like [41(1)] — that is invalid JSON.
2. All string values MUST use double quotes, never single quotes.
3. Return ONLY the JSON object. No preamble, no explanation, no markdown.
4. The response MUST end with the closing brace }}.
5. "reasoning" must be ONE short sentence. Do NOT write paragraphs.

CLASSIFICATION RULES:
1. If the user is asking for information/explanation only → is_actionable: false
2. If the user wants to DO something → is_actionable: true
3. If multiple actions are equally possible → ambiguous: true, pick most likely
4. confidence < 0.5 → set ambiguous: true
5. NEVER invent action codes outside the taxonomy
6. NEVER include form names or URLs in your output
7. Respond in the same language classification but always return JSON in English keys

TAXONOMY DIRECTION NOTE:
  convert_public_to_private  = a PUBLIC company becoming PRIVATE (s.41(1))
  convert_private_to_public  = a PRIVATE company becoming PUBLIC (s.41(2))
  These are OPPOSITE directions. Read the user query carefully before classifying.
"""


# ─── Agent implementation ──────────────────────────────────────────────────────

class IntentFormAgent:
    """
    LLM-powered intent classifier that maps natural language queries
    to structured legal actions, which are then bound to forms deterministically.

    Design principles:
    - Agent scope is NARROW: intent classification only
    - Agent output is STRUCTURED: fixed schema, validated
    - Form binding is DETERMINISTIC: agent never touches form names
    - Fallback is GRACEFUL: if agent fails, falls back to section matching
    """

    # Ollama endpoint (reuses AppConfig)
    _OLLAMA_URL = "http://localhost:11434/api/chat"

    # Use a fast small model for classification — 
    # intent classification does not need the full 8B model
    _CLASSIFIER_MODEL = "qwen3:1.7b"

    # Hard timeout — this runs inline in the query pipeline
    _TIMEOUT = 15

    # Confidence threshold below which we fall back to section matching
    _MIN_CONFIDENCE = 0.55

    def __init__(self, forms_data: List[Dict]) -> None:
        self._forms = self._build_form_index(forms_data)
        self._taxonomy_str = self._build_taxonomy_string()
        self._cache: Dict[str, FormResolutionResult] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def _load_form_overrides(self) -> Dict[str, Dict]:
        """Read corrections written by LearningAgent.CorrectionEngine."""
        if not os.path.exists(_FORM_OVERRIDES_FILE):
            return {}
        try:
            with open(_FORM_OVERRIDES_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    @staticmethod
    def _load_query_intent_corrections() -> Dict[str, str]:
        """
        Returns {query_hash: correct_intent_action}.
        Written by CorrectionEngine when thumbs-down + self-reflection produces
        a suggested_intent.  Allows resolve() to hard-route the query to the
        correct intent without calling the LLM classifier at all.
        """
        if not os.path.exists(_QUERY_INTENT_FILE):
            return {}
        try:
            with open(_QUERY_INTENT_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    @staticmethod
    def _load_classifier_patches() -> str:
        """
        Reads learned_patches.json and assembles a compact correction block
        to prepend to the classifier system prompt.
        Only includes intent-direction-correction entries so the classifier
        is not polluted with generation-level patches.
        """
        if not os.path.exists(_CLASSIFIER_PATCHES_FILE):
            return ""
        try:
            with open(_CLASSIFIER_PATCHES_FILE, "r", encoding="utf-8") as fh:
                patches = json.load(fh)
        except Exception:
            return ""

        lines = []
        for entry in patches.values():
            if isinstance(entry, dict) and entry.get("type") == "intent_direction_correction":
                lines.append(entry.get("patch_text", ""))

        if not lines:
            return ""
        return (
            "\nCORRECTIONS FROM USER FEEDBACK (apply these first):\n"
            + "\n".join(f"  • {ln}" for ln in lines)
            + "\n"
        )

    def resolve(
        self,
        query: str,
        retrieved_sections: Dict[tuple, int],
        conversation_history: List[Dict],
        lang: str = "en",
        detected_act: Optional[str] = None,
        uid: Optional[str] = None,
    ) -> FormResolutionResult:
        """
        Main entry point. Returns FormResolutionResult.

        Resolution priority:
        1. Agent intent classification (if actionable and high confidence)
        2. Section-frequency matching from retrieved chunks (fallback)
        3. Conversation context carry-forward (for follow-ups)
        4. Empty result with explicit reason
        """

        # Cache hit (same query in same session)
        cache_key = hashlib.md5(f"{uid}:{query[:100]}_{lang}".encode()).hexdigest()
        if cache_key in _INTENT_CACHE and cache_key not in _NEGATIVELY_RATED:
            logger.debug("IntentFormAgent: cache hit for query.")
            return _INTENT_CACHE[cache_key]

        # ── Step 0: Check stored query-hash → correct-intent corrections ──────
        # Written by CorrectionEngine after thumbs-down + self-reflection.
        # Bypasses the LLM entirely for queries that have been previously
        # misclassified and corrected — no risk of re-making the same mistake.
        query_hash = hashlib.md5(
            re.sub(r"\s+", " ", query.lower().strip()).rstrip("?.! ").encode()
        ).hexdigest()[:8]
        intent_corrections = self._load_query_intent_corrections()
        if query_hash in intent_corrections and cache_key in _NEGATIVELY_RATED:
            correct_action = intent_corrections[query_hash]
            logger.info(
                "IntentFormAgent: using stored intent correction for hash=%s → '%s'",
                query_hash, correct_action,
            )
            # Synthesize a LegalIntent using the stored correct action
            intent = LegalIntent(
                action          = correct_action,
                legal_procedure = correct_action.replace("_", " ").title(),
                likely_sections = INTENT_TAXONOMY.get(correct_action, []),
                confidence      = 1.0,
                reasoning       = "Stored correction from user feedback",
                language        = lang,
                is_actionable   = True,
                ambiguous       = False,
            )
            result = self._resolve_from_intent(intent, detected_act=detected_act)
            if result.forms:
                if len(_INTENT_CACHE) >= _INTENT_CACHE_MAX:
                    _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)))
                _INTENT_CACHE[cache_key] = result
                _NEGATIVELY_RATED.discard(cache_key)   # correction applied — reset
                return result

        # Step 1: Run agent classification (with any learned patches injected)
        intent = self._classify_intent(query, conversation_history, lang)

        if intent and intent.action != "unknown":
            overrides = self._load_form_overrides()
            override  = overrides.get(intent.action)
            if override and override.get("locked") and override.get("form_id"):
            # Find the form in the index by form_id
                locked_form = next(
                    (f for f in self._forms if f.get("form_id") == override["form_id"]),
                    None,
                )
                if locked_form:
                    logger.info(
                        "IntentFormAgent: using locked override for action '%s' → '%s'",
                        intent.action, override.get("form_name", ""),
                    )
                    result = FormResolutionResult(
                        forms=[self._to_output_dict(locked_form)],
                        intent=intent,
                        resolution_path="locked_override",
                        confidence=1.0,
                    )
                    if len(_INTENT_CACHE) >= _INTENT_CACHE_MAX:
                        _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)))
                    _INTENT_CACHE[cache_key] = result
                    return result

        # Step 2: Route based on intent result
        if intent and intent.is_actionable and intent.confidence >= self._MIN_CONFIDENCE and not intent.ambiguous:
            result = self._resolve_from_intent(intent, detected_act=detected_act)
            if result.forms:
                logger.info(
                    "IntentFormAgent: resolved via agent intent '%s' "
                    "(confidence=%.2f) → %d form(s)",
                    intent.action, intent.confidence, len(result.forms)
                )
                if len(_INTENT_CACHE) >= _INTENT_CACHE_MAX:
                    _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)))
                _INTENT_CACHE[cache_key] = result
                return result

        # Step 3: Fall back to section matching
        if retrieved_sections:
            result = self._resolve_from_sections(retrieved_sections, intent)
            if result.forms:
                logger.info(
                    "IntentFormAgent: resolved via section matching → %d form(s)",
                    len(result.forms)
                )
                if len(_INTENT_CACHE) >= _INTENT_CACHE_MAX:
                    _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)))
                _INTENT_CACHE[cache_key] = result
                return result

        # Step 4: Non-actionable or no match
        reason = "not_actionable" if (intent and not intent.is_actionable) else "no_match"
        result = FormResolutionResult(
            forms=[], intent=intent,
            resolution_path="none", confidence=0.0
        )

        if len(_INTENT_CACHE) >= _INTENT_CACHE_MAX:
            _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)))  # evict oldest
        _INTENT_CACHE[cache_key] = result

        logger.info(
            "IntentFormAgent.resolve: action='%s' confidence=%.2f path='%s' forms=%d",
            intent.action if intent else "none",
            intent.confidence if intent else 0.0,
            result.resolution_path,
            len(result.forms),
        )

        return result
    
    def invalidate(self, query: str, lang: str = "en") -> None:
        """Called by LearningAgent when a query receives thumbs down."""
        cache_key = hashlib.md5(f"{query[:100]}_{lang}".encode()).hexdigest()
        _NEGATIVELY_RATED.add(cache_key)
        _INTENT_CACHE.pop(cache_key, None)
        logger.info("IntentFormAgent: invalidated cache for negatively rated query.")

    def clear_negative_rating(self, query: str, lang: str = "en") -> None:
        """Called on thumbs-up to re-enable caching for this query."""
        cache_key = hashlib.md5(f"{query[:100]}_{lang}".encode()).hexdigest()
        _NEGATIVELY_RATED.discard(cache_key)
        logger.info("IntentFormAgent: cleared negative rating flag for query.")

    # ── Agent classification ───────────────────────────────────────────────────

    def _classify_intent(
        self,
        query: str,
        conversation_history: List[Dict],
        lang: str,
    ) -> Optional[LegalIntent]:
        """
        Calls the LLM with a narrow classification prompt.
        Returns None on any failure — fallback handles it.
        """

        # Build minimal conversation context (last 2 turns only)
        context_lines = []
        for turn in conversation_history[-2:]:
            context_lines.append(f"User: {turn.get('query', '')[:200]}")
            context_lines.append(f"Assistant summary: {turn.get('summary', '')[:150]}")
        context_block = "\n".join(context_lines) if context_lines else "None"

        user_message = (
            f"Conversation context (last 2 turns):\n{context_block}\n\n"
            f"Current user query: {query}"
        )

        # Inject any learned classifier patches (intent-direction corrections
        # written by CorrectionEngine after previous thumbs-down events).
        classifier_patches = self._load_classifier_patches()
        system = _AGENT_SYSTEM_PROMPT.format(taxonomy=self._taxonomy_str)
        if classifier_patches:
            # Prepend corrections so the model sees them at max attention weight
            system = classifier_patches + "\n" + system
            logger.debug("IntentFormAgent: injected %d classifier patch(es).", classifier_patches.count("•"))

        try:
            resp = requests.post(
                self._OLLAMA_URL,
                json={
                    "model": self._CLASSIFIER_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user_message},
                    ],
                    "stream": False,
                    "think":  False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 180,
                        "num_ctx":     2048,
                        "stop": ["\n}\n", "}\n"],
                    },
                },
                timeout=(5, self._TIMEOUT),
            )

            if resp.status_code != 200:
                logger.warning(
                    "IntentFormAgent: classifier returned HTTP %d", resp.status_code
                )
                return None

            raw = resp.json().get("message", {}).get("content", "").strip()

            # Strip any thinking blocks
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

            return self._parse_intent(raw, lang)

        except requests.Timeout:
            logger.warning(
                "IntentFormAgent: classifier timed out after %ds — "
                "falling back to section matching.", self._TIMEOUT
            )
            return None
        except Exception as exc:
            logger.warning("IntentFormAgent: classification failed: %s", exc)
            return None

    def _parse_intent(self, raw_json: str, lang: str) -> Optional[LegalIntent]:
        """Parse and validate the agent's JSON output."""
        try:
            # ── Repair Pass 1: fix unquoted section numbers like [41(1), 58] ────
            # Matches array values that look like section numbers but are unquoted
            repaired = re.sub(
                r'\[([^\]"]*?)\]',   # find array content
                lambda m: '[' + ', '.join(
                    f'"{item.strip()}"' if item.strip() and not item.strip().startswith('"')
                    else item.strip()
                    for item in m.group(1).split(',')
                    if item.strip()
                ) + ']',
                raw_json,
            )

            # ── Repair Pass 2: close truncated JSON ──────────────────────────────
            # If the JSON is missing its closing brace, add it
            stripped = repaired.strip()
            if stripped.startswith('{') and not stripped.endswith('}'):
                # Find the last complete key-value pair by looking for the last comma
                # or the last complete quoted string
                last_complete = max(
                    stripped.rfind('",\n'),
                    stripped.rfind('",'),
                    stripped.rfind('",\r\n'),
                )
                if last_complete != -1:
                    # Truncate at the last complete pair and close the object
                    repaired = stripped[:last_complete + 1] + '\n}'
                else:
                    repaired = stripped + '\n}'
                logger.warning(
                    "IntentFormAgent: repaired truncated JSON (added closing brace)."
                )

            data = json.loads(repaired)

            action = data.get("action", "unknown")
            if action != "unknown" and action not in INTENT_TAXONOMY:
                logger.warning(
                    "IntentFormAgent: agent returned unknown action '%s' — treating as unknown",
                    action
                )
                action = "unknown"

            return LegalIntent(
                action          = action,
                legal_procedure = str(data.get("legal_procedure", ""))[:200],
                likely_sections = [
                    str(s) for s in data.get("likely_sections", [])[:6]
                ],
                confidence      = float(data.get("confidence", 0.0)),
                reasoning       = str(data.get("reasoning", ""))[:300],
                language        = lang,
                is_actionable   = bool(data.get("is_actionable", False)),
                ambiguous       = bool(data.get("ambiguous", False)),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "IntentFormAgent: could not parse agent output: %s | raw: %s",
                exc, raw_json[:200]
            )
            return None

    # ── Form resolution from intent ────────────────────────────────────────────

    def _resolve_from_intent(self, intent: LegalIntent, detected_act: Optional[str] = None,) -> FormResolutionResult:
        """
        Maps a classified intent action to forms.
        Uses the taxonomy's section list to match forms.json entries.
        100% deterministic — no LLM involved.
        """
        if intent.action == "unknown":
            return FormResolutionResult(
                forms=[], intent=intent,
                resolution_path="agent", confidence=0.0
            )

        # Get expected sections from taxonomy
        expected_sections = INTENT_TAXONOMY.get(intent.action, [])
        if not expected_sections:
            return FormResolutionResult(
                forms=[], intent=intent,
                resolution_path="agent", confidence=intent.confidence
            )

        # Also include sections the agent itself identified
        all_sections = set(expected_sections)

        matched_forms = []
        for form_entry in self._forms:
            form_sections = form_entry.get("_parsed_sections", [])
            # Score: how many of the expected sections match this form
            score = sum(
                1 for es in all_sections
                for fs in form_sections
                if self._sections_match(es, fs)
            )
            if score > 0:
                matched_forms.append((score, form_entry))

        if detected_act and matched_forms:
            act_filtered = [
                (score, f) for score, f in matched_forms
                if not f.get("_act_filter") or detected_act in f.get("_act_filter", [])
            ]
            if act_filtered:
                matched_forms = act_filtered

        matched_forms.sort(key=lambda x: x[0], reverse=True)

        # Return top 3 forms maximum
        top_forms = [
            self._to_output_dict(f)
            for _, f in matched_forms[:3]
        ]

        return FormResolutionResult(
            forms=top_forms,
            intent=intent,
            resolution_path="agent",
            confidence=intent.confidence,
        )

    # ── Section-frequency fallback ─────────────────────────────────────────────

    def _resolve_from_sections(
        self,
        section_freq: Dict[tuple, int],
        intent: Optional[LegalIntent],
    ) -> FormResolutionResult:
        """
        Deterministic fallback: match forms by dominant retrieved sections.
        Used when agent confidence is low or agent fails.
        """
        if not section_freq:
            return FormResolutionResult(
                forms=[], intent=intent,
                resolution_path="fallback", confidence=0.0
            )

        total = sum(section_freq.values())
        max_freq = max(section_freq.values())
        scored = []

        for form_entry in self._forms:
            form_sections = form_entry.get("_parsed_sections", [])
            score = 0.0
            form_best_freq = 0

            for ref, freq in section_freq.items():
                ref_str = ref[0] + (f"({ref[1]})" if ref[1] else "")
                for fs in form_sections:
                    if self._sections_match(ref_str, fs):
                        # Weight by relative frequency
                        score += freq / total
                        form_best_freq = max(form_best_freq, freq)

            if score == 0:
                continue

            # ── Dominance gate ────────────────────────────────────────────────────
            # A form is only returned when its linked section is actually dominant
            # in the retrieved context — not just incidentally mentioned once.
            #
            # Gate 1: raw score must exceed 0.30 (was 0.15)
            #   → requires the section to represent ≥30% of all section mentions
            # Gate 2: the section must appear in at least 60% as many chunks as
            #   the most-cited section in the results
            #   → blocks forms tied to incidental sections (Section 58 appearing
            #     once while Section 46 appears 4 times is now blocked)
            if score < 0.30:
                continue
            if form_best_freq < max_freq * 0.60:
                logger.debug(
                    "_resolve_from_sections: skipping '%s' — "
                    "section freq %d < 60%% of dominant %d",
                    form_entry.get("name", "")[:40], form_best_freq, max_freq,
                )
                continue

            scored.append((score, form_entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_forms = [self._to_output_dict(f) for _, f in scored[:3]]

        return FormResolutionResult(
            forms=top_forms,
            intent=intent,
            resolution_path="section_match",
            confidence=max((s for s, _ in scored[:1]), default=0.0),
        )

    # ── Section matching utility ───────────────────────────────────────────────

    @staticmethod
    def _sections_match(query_sec: str, form_sec: str) -> bool:
        """
        Flexible section matching that handles format variations:
        "41(2)" matches "Section 41(2)", "section 41(2)", "s41(2)"
        """
        # Normalize both to bare number format
        def normalize(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r'^(section|seksyen|s\.?)\s*', '', s)
            s = re.sub(r'\s+', '', s)
            return s

        return normalize(query_sec) == normalize(form_sec)

    # ── Index building ─────────────────────────────────────────────────────────

    def _build_form_index(self, forms_data: List[Dict]) -> List[Dict]:
        """
        Pre-process forms.json into a queryable index.
        Parses related_sections into normalized tuples for fast matching.
        """
        indexed = []
        _ACT_MAP = {
            "companies act": "Companies Act 2016",
            "act 777":       "Companies Act 2016",
            "llp act":       "LLP Act 2012",
            "act 743":       "LLP Act 2012",
            "rob act":       "Registration of Businesses Act 1956",
            "act 197":       "Registration of Businesses Act 1956",
        }
        _SEC_EXTRACT_RE = re.compile(
            r'(?:section|seksyen|s\.?)\s*(\d+[a-z]?)(?:\((\w+)\))?',
            re.IGNORECASE
        )

        for form in forms_data:
            parsed_sections = []
            for rs in form.get("related_sections", []):
                # Extract bare section number
                m = _SEC_EXTRACT_RE.search(rs)
                if m:
                    parent = m.group(1).lower()
                    child  = m.group(2).lower() if m.group(2) else None
                    parsed_sections.append(
                        parent + (f"({child})" if child else "")
                    )

            act_filter = []
            for rs in form.get("related_sections", []):
                rs_lower = rs.lower()
                for keyword, act in _ACT_MAP.items():
                    if keyword in rs_lower and act not in act_filter:
                        act_filter.append(act)
            # Default: if no act found, assume Companies Act (most common)
            if not act_filter:
                act_filter = ["Companies Act 2016"]
            
            form_copy = dict(form)
            form_copy["_parsed_sections"] = parsed_sections
            form_copy["_act_filter"] = act_filter
            indexed.append(form_copy)
        return indexed

    def _build_taxonomy_string(self) -> str:
        lines = []
        for action, sections in INTENT_TAXONOMY.items():
            lines.append(f"  {action} (sections: {', '.join(sections)})")
        return "\n".join(lines)

    @staticmethod
    def _to_output_dict(form: Dict) -> Dict:
        links = form.get("links", [])
        primary_link = next(
            (l for l in links if l.get("type") in ("pdf", "portal", "platform")),
            {}
        )
        return {
            "name":        form["name"],
            "form_number": form.get("form_number", ""),
            "url":           primary_link.get("url", ""),
            "link_type":     primary_link.get("type", "pdf"),
            "resource_type": form.get("resource_type", "form"),
        }