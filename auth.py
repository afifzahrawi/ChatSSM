# auth.py
"""
Authentication layer for ChatSSM.
Replaces URL-UUID identity with Supabase-backed login.
Drop this file next to chatssm_app.py.
"""

import os
import streamlit as st
from supabase import create_client, Client

_SUPABASE_URL = os.environ["SUPABASE_URL"]
_SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]


@st.cache_resource
def _supabase() -> Client:
    return create_client(_SUPABASE_URL, _SUPABASE_KEY)


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
                            st.success("Account created! Check your email to confirm, then log in.")
                        else:
                            st.error("Sign-up failed — please try again.")
                    except Exception as exc:
                        st.error(f"Sign-up failed: {exc}")

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