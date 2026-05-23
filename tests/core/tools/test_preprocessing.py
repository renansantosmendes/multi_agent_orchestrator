from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from src.core.tools import store
from src.core.tools.preprocessing import preprocess_data


def _make_fetal_health_df() -> pd.DataFrame:
    """Returns a minimal DataFrame that mimics the fetal_health dataset structure."""
    rng = np.random.default_rng(42)
    n = 50
    data = {f"feature_{i}": rng.random(n) for i in range(5)}
    data["fetal_health"] = rng.choice([1.0, 2.0, 3.0], size=n)
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def reset_store() -> None:
    """Clears the pipeline_store before and after every test."""
    store.pipeline_store.clear()
    yield
    store.pipeline_store.clear()


class TestPreprocessData:
    """Tests for the preprocess_data tool."""

    @patch("src.core.tools.preprocessing.pd.read_csv")
    def test_returns_command(self, mock_read_csv) -> None:
        """Verifies that the tool returns a Command instance."""
        mock_read_csv.return_value = _make_fetal_health_df()

        result = preprocess_data.func(tool_call_id="test-id")

        assert isinstance(result, Command)

    @patch("src.core.tools.preprocessing.pd.read_csv")
    def test_populates_pipeline_store(self, mock_read_csv) -> None:
        """Verifies that the pipeline_store is populated with train/test splits."""
        mock_read_csv.return_value = _make_fetal_health_df()

        preprocess_data.func(tool_call_id="test-id")

        assert "X_train" in store.pipeline_store
        assert "X_test" in store.pipeline_store
        assert "y_train" in store.pipeline_store
        assert "y_test" in store.pipeline_store
        assert "scaler" in store.pipeline_store
        assert "feature_cols" in store.pipeline_store

    @patch("src.core.tools.preprocessing.pd.read_csv")
    def test_command_update_flags_preprocessing_done(self, mock_read_csv) -> None:
        """Verifies the Command update marks preprocessing as done."""
        mock_read_csv.return_value = _make_fetal_health_df()

        result = preprocess_data.func(tool_call_id="test-id")

        assert result.update["preprocessing_done"] is True
        assert result.update["current_stage"] == "preprocessing"

    @patch("src.core.tools.preprocessing.pd.read_csv")
    def test_target_column_excluded_from_features(self, mock_read_csv) -> None:
        """Verifies the target column is not included in feature_columns."""
        mock_read_csv.return_value = _make_fetal_health_df()

        result = preprocess_data.func(target_column="fetal_health", tool_call_id="test-id")

        assert "fetal_health" not in result.update["feature_columns"]

    @patch("src.core.tools.preprocessing.pd.read_csv")
    def test_summary_contains_key_info(self, mock_read_csv) -> None:
        """Verifies the summary message contains feature and sample counts."""
        mock_read_csv.return_value = _make_fetal_health_df()

        result = preprocess_data.func(tool_call_id="test-id")

        summary = result.update["messages"][0].content
        assert "PREPROCESSING COMPLETE" in summary
        assert "Features:" in summary
        assert "Train:" in summary
        assert "Test:" in summary
