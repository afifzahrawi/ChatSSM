# auth.py
"""
Authentication layer for ChatSSM.
Replaces URL-UUID identity with Supabase-backed login.
Drop this file next to chatssm_app.py.
"""

import os
import streamlit as st

# Monkey-patch httpx to disable SSL verification (Windows SSL workaround)
import httpx

class PatchedClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)

class PatchedAsyncClient(httpx.AsyncClient):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)

httpx.Client = PatchedClient
httpx.AsyncClient = PatchedAsyncClient

from supabase import create_client, Client

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
if not _SUPABASE_URL or not _SUPABASE_KEY:
    raise EnvironmentError(
        "SUPABASE_URL and SUPABASE_ANON_KEY must be set as environment variables."
    )

@st.cache_resource
def _supabase() -> Client:
    return create_client(_SUPABASE_URL, _SUPABASE_KEY)

@st.cache_resource
def _supabase_admin() -> Client:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY", "")
    if not key:
        raise EnvironmentError("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY must be set.")
    return create_client(os.environ["SUPABASE_URL"], key)


def render_auth_wall() -> str | None:
    """
    Renders login / signup UI and returns the authenticated user_id,
    or None if the user is not yet logged in.

    Call this at the TOP of main() before any other UI code.
    Returns user_id string on success, None if not authenticated.
    """
    # Already logged in this session
    if st.session_state.get("_auth_user_id"):
        return st.session_state["_auth_user_id"]

    st.markdown(
        """
        <div style="
            max-width:380px; margin:80px auto 0;
            text-align:center; padding:0 20px;
        ">
            <h1 style="font-size:2rem; font-weight:700; margin-bottom:6px;">⚖️ ChatSSM</h1>
            <p style="color:#888; margin-bottom:28px;">
                Your AI assistant for Malaysian company law
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

        with tab_login:
            email    = st.text_input("Email",    key="login_email",    placeholder="you@example.com")
            password = st.text_input("Password", key="login_password", type="password", placeholder="••••••••")
            if st.button("Log in", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    try:
                        res = _supabase().auth.sign_in_with_password(
                            {"email": email, "password": password}
                        )
                        if res.session is None or res.user is None:
                            st.error("Login failed: Email confirmation required. Please check your inbox.")
                        else:
                            _set_session(res.user.id, res.session.access_token)
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Login failed: {exc}")

        with tab_signup:
            email    = st.text_input("Email",            key="signup_email",    placeholder="you@example.com")
            password = st.text_input("Password",         key="signup_password", type="password", placeholder="Min 8 characters")
            confirm  = st.text_input("Confirm password", key="signup_confirm",  type="password")

            if st.session_state.get("_signup_success"):
                st.success("✅ Account created! You can now log in.")
                st.session_state.pop("_signup_success", None)

            if st.button("Create account", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Please fill in all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    try:
                        res = _supabase().auth.sign_up({"email": email, "password": password})
                        if res.user:
                            st.session_state["_signup_success"] = True
                            st.rerun()
                        else:
                            st.error("Sign-up failed — please try again.")
                    except Exception as exc:
                        msg = str(exc)
                        if "already registered" in msg.lower() or "already exists" in msg.lower():
                            st.error("This email is already registered. Please log in instead.")
                        else:
                            st.error(f"Sign-up failed: {msg}")

    return None  # not authenticated yet


def logout() -> None:
    """Call when the user clicks Log out."""
    try:
        _supabase().auth.sign_out()
    except Exception:
        pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _set_session(user_id: str, access_token: str) -> None:
    st.session_state["_auth_user_id"]      = user_id
    st.session_state["_auth_access_token"] = access_token