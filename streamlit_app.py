"""Streamlit Community Cloud entry point.

Adds the repo root to sys.path so `stock_forecasting.*` imports resolve without
installing the package, then runs the app.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from stock_forecasting.app import main

main()
