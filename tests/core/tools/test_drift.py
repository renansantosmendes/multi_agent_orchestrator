from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langgraph.types import Command

from src.core.tools.drift import detect_data_drift


def _make_dataframe() -> pd.DataFrame:
    """Returns a minimal DataFrame that mimics the fetal_health dataset."""
    return pd.DataFrame({"col1": [1.0, 2.0, 3.0, 4.0], "col2": [0.1, 0.2, 0.3, 0.4]})


def _make_report_result(drift_count: int, drift_share: float, col_p_values: list) -> MagicMock:
    """Builds a mock Evidently result dict with the given drift statistics."""
    metrics = [{"value": {"count": drift_count, "share": drift_share}}]
    for col, p_val in col_p_values:
        metrics.append({"config": {"column": col, "threshold": 0.05}, "value": p_val})
    mock_result = MagicMock()
    mock_result.dict.return_value = {"metrics": metrics}
    return mock_result


class TestDetectDataDrift:
    """Tests for the detect_data_drift tool."""

    @patch("src.core.tools.drift.Report")
    @patch("src.core.tools.drift.pd.read_csv")
    def test_returns_command(self, mock_read_csv: MagicMock, mock_report_class: MagicMock) -> None:
        """Verifies that the tool always returns a Command instance."""
        mock_read_csv.return_value = _make_dataframe()
        mock_report_class.return_value.run.return_value = _make_report_result(
            drift_count=0, drift_share=0.0, col_p_values=[("col1", 0.1), ("col2", 0.2)]
        )

        result = detect_data_drift.func(
            reference_start=0, reference_end=2,
            current_start=2, current_end=4,
            tool_call_id="test-id",
        )

        assert isinstance(result, Command)

    @patch("src.core.tools.drift.Report")
    @patch("src.core.tools.drift.pd.read_csv")
    def test_no_drift_when_share_below_threshold(
        self, mock_read_csv: MagicMock, mock_report_class: MagicMock
    ) -> None:
        """Verifies drift_detected is False when drift share is below the threshold."""
        mock_read_csv.return_value = _make_dataframe()
        mock_report_class.return_value.run.return_value = _make_report_result(
            drift_count=0, drift_share=0.2, col_p_values=[("col1", 0.1), ("col2", 0.2)]
        )

        result = detect_data_drift.func(
            reference_start=0, reference_end=2,
            current_start=2, current_end=4,
            drift_share_threshold=0.5,
            tool_call_id="test-id",
        )

        assert result.update["drift_detected"] is False
        assert result.update["current_stage"] == "drift_check"

    @patch("src.core.tools.drift.Report")
    @patch("src.core.tools.drift.pd.read_csv")
    def test_drift_detected_when_share_at_or_above_threshold(
        self, mock_read_csv: MagicMock, mock_report_class: MagicMock
    ) -> None:
        """Verifies drift_detected is True when drift share meets the threshold."""
        mock_read_csv.return_value = _make_dataframe()
        mock_report_class.return_value.run.return_value = _make_report_result(
            drift_count=2, drift_share=0.8, col_p_values=[("col1", 0.01), ("col2", 0.02)]
        )

        result = detect_data_drift.func(
            reference_start=0, reference_end=2,
            current_start=2, current_end=4,
            drift_share_threshold=0.5,
            tool_call_id="test-id",
        )

        assert result.update["drift_detected"] is True
        assert "col1" in result.update["drifted_columns"]
        assert "col2" in result.update["drifted_columns"]

    @patch("src.core.tools.drift.Report")
    @patch("src.core.tools.drift.pd.read_csv")
    def test_summary_contains_report_header(
        self, mock_read_csv: MagicMock, mock_report_class: MagicMock
    ) -> None:
        """Verifies the summary message contains the expected header."""
        mock_read_csv.return_value = _make_dataframe()
        mock_report_class.return_value.run.return_value = _make_report_result(
            drift_count=0, drift_share=0.0, col_p_values=[]
        )

        result = detect_data_drift.func(
            reference_start=0, reference_end=2,
            current_start=2, current_end=4,
            tool_call_id="test-id",
        )

        summary = result.update["drift_summary"]
        assert "DATA DRIFT REPORT" in summary
        assert "GLOBAL DRIFT" in summary
