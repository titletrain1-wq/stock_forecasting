"""Streamlit Community Cloud entry point.

Adds the repo root to sys.path so `stock_forecasting.*` imports resolve without
installing the package, bridges Streamlit secrets into the environment (the
service layer reads config via env vars / pydantic-settings, not st.secrets),
then runs the app.
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

# Streamlit Cloud exposes secrets via st.secrets; copy them into os.environ so
# pydantic-settings (env-var based) picks up TURSO_DATABASE_URL / TURSO_AUTH_TOKEN.
try:  # pragma: no cover - only meaningful on Streamlit Cloud
    import streamlit as st

    for _k, _v in dict(st.secrets).items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:  # noqa: BLE001
    pass

from stock_forecasting.app import main  # noqa: E402

main()
