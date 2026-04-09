# db_storage.py
"""
Supabase-backed replacement for StorageService.
Same method signatures — swap it in without changing chatssm_app.py's call sites.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from auth import _supabase

logger = logging.getLogger("chatssm.db")


class DBStorageService:

    # ── Sessions ──────────────────────────────────────────────────────────────

    @staticmethod
    def list_sessions(uid: str) -> List[Dict]:
        try:
            response = (
                _supabase()
                .table("chat_sessions")
                .select("id, title, updated_at")
                .eq("user_id", uid)
                .order("updated_at", desc=True)
                .execute()
            )
            raw_data = response.data
            if not isinstance(raw_data, list):
                return []
            rows = [r for r in raw_data if isinstance(r, dict)]
            # Normalise to match the shape StorageService returned
            return [{"id": r["id"], "title": r["title"], "ts": r["updated_at"]} for r in rows]
        except Exception as exc:
            logger.error("list_sessions failed: %s", exc)
            return []

    @staticmethod
    def create_session(uid: str) -> str:
        try:
            response = (
                _supabase()
                .table("chat_sessions")
                .insert({"user_id": uid, "title": "New Chat"})
                .execute()
            )
            raw_data = response.data
            if not isinstance(raw_data, list) or len(raw_data) == 0:
                raise ValueError("Insert returned no data")
            row = raw_data[0]
            if not isinstance(row, dict):
                raise ValueError("Insert returned non-dict row")
            return str(row.get("id", ""))
        except Exception as exc:
            logger.error("create_session failed: %s", exc)
            import uuid
            return uuid.uuid4().hex[:8]   # fallback

    @staticmethod
    def load_session(uid: str, session_id: str) -> List[Dict]:
        try:
            raw_data = (
                _supabase()
                .table("messages")
                .select("*")
                .eq("session_id", session_id)
                .eq("user_id", uid)          # double-check ownership even with RLS
                .order("created_at", desc=False)
                .execute()
                .data
            )
            if not isinstance(raw_data, list):
                return []
            rows = [r for r in raw_data if isinstance(r, dict)]
            # Convert DB rows → the dict shape chatssm_app.py expects
            return [
                {
                    "qa_id":          r.get("qa_id", ""),
                    "query":          r["query"],
                    "response":       r["response"],
                    "lang":           r.get("lang", "en"),
                    "citations":      r.get("citations") or [],
                    "categories_hit": r.get("categories_hit") or [],
                    "forms":          r.get("forms") or [],
                    "timestamp":      r["created_at"],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.error("load_session failed: %s", exc)
            return []

    @staticmethod
    def save_session(uid: str, session_id: str, history: List[Dict]) -> None:
        """
        Append only the messages that aren't yet in DB.
        We use qa_id as the idempotency key — safe to call repeatedly.
        """
        if not history:
            return
        try:
            # Find which qa_ids already exist for this session
            raw_data = (
                _supabase()
                .table("messages")
                .select("qa_id")
                .eq("session_id", session_id)
                .execute()
                .data
            )
            if not isinstance(raw_data, list):
                existing = set()
            else:
                existing = set(
                    r["qa_id"]
                    for r in raw_data
                    if isinstance(r, dict) and r.get("qa_id")
                )

            new_rows = []
            for msg in history:
                if msg.get("qa_id") and msg["qa_id"] in existing:
                    continue
                new_rows.append({
                    "session_id":      session_id,
                    "user_id":         uid,
                    "qa_id":           msg.get("qa_id", ""),
                    "query":           msg["query"],
                    "response":        msg["response"],
                    "lang":            msg.get("lang", "en"),
                    "citations":       msg.get("citations", []),
                    "categories_hit":  msg.get("categories_hit", []),
                    "forms":           msg.get("forms", []),
                })

            if new_rows:
                _supabase().table("messages").upsert(
                    new_rows, on_conflict="qa_id,session_id"
                ).execute()

            # Update session title from first message
            if history:
                first_q = history[0].get("query", "")
                title   = (first_q[:52] + "…") if len(first_q) > 52 else first_q or "New Chat"
                (
                    _supabase()
                    .table("chat_sessions")
                    .update({"title": title, "updated_at": datetime.now().isoformat()})
                    .eq("id", session_id)
                    .execute()
                )
        except Exception as exc:
            logger.error("save_session failed: %s", exc)

    @staticmethod
    def delete_session(uid: str, session_id: str) -> None:
        try:
            _supabase().table("chat_sessions").delete()\
                .eq("id", session_id).eq("user_id", uid).execute()
        except Exception as exc:
            logger.error("delete_session failed: %s", exc)

    # ── Shims matching StorageService's call signatures ───────────────────────

    @staticmethod
    def load_history(uid: str) -> List[Dict]:
        import streamlit as st
        sid = st.session_state.get("active_session_id")
        return DBStorageService.load_session(uid, sid) if sid else []

    @staticmethod
    def save_history(history: List[Dict], uid: str) -> None:
        import streamlit as st
        sid = st.session_state.get("active_session_id")
        if sid:
            DBStorageService.save_session(uid, sid, history)

    @staticmethod
    def log_qa(query, answer, sources, rating=None, qa_id=None):
        # Retained for compatibility — messages table already stores this data
        pass