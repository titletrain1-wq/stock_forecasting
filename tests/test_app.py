import sys
from unittest.mock import MagicMock

import pytest

sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit_lightweight_charts'] = MagicMock()

def test_app_imports_cleanly():
    try:
        import stock_forecasting.app
        assert hasattr(stock_forecasting.app, 'main')
    except ImportError as e:
        pytest.fail(f"app.py raised exception on import: {e}")
