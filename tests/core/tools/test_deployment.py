from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

import pytest
from langgraph.types import Command
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.core.tools import store
from src.core.tools.deployment import deploy_model


def _populate_store_with_trained_model() -> None:
    """Trains a small model and populates the pipeline_store with all required keys."""
    X, y = make_classification(n_samples=60, n_features=5, n_classes=3,
                                n_informative=3, n_redundant=2, random_state=42)
    y = y.astype(float) + 1.0
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X[:50], y[:50])
    scaler = StandardScaler()
    scaler.fit(X[:50])

    store.pipeline_store.update({
        "trained_model": model,
        "scaler": scaler,
        "last_model_name": "RandomForest",
        "last_accuracy": 0.9,
        "feature_cols": [f"f{i}" for i in range(4)],
    })


@pytest.fixture(autouse=True)
def reset_store() -> None:
    """Clears the pipeline_store before and after every test."""
    store.pipeline_store.clear()
    yield
    store.pipeline_store.clear()


class TestDeployModel:
    """Tests for the deploy_model tool."""

    def test_error_when_no_model_in_store(self) -> None:
        """Verifies an error message is returned when no trained model exists."""
        result = deploy_model.func(tool_call_id="test-id")

        assert isinstance(result, Command)
        assert "Error" in result.update["messages"][0].content

    @patch("src.core.tools.deployment.joblib.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_command_on_success(self, mock_file, mock_dump) -> None:
        """Verifies a Command is returned when deployment succeeds."""
        _populate_store_with_trained_model()

        result = deploy_model.func(model_path="model.joblib", tool_call_id="test-id")

        assert isinstance(result, Command)
        assert result.update["current_stage"] == "deployed"
        assert result.update["model_path"] == "model.joblib"

    @patch("src.core.tools.deployment.joblib.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_joblib_dump_called_for_model_and_scaler(self, mock_file, mock_dump) -> None:
        """Verifies joblib.dump is called for both the model and the scaler."""
        _populate_store_with_trained_model()

        deploy_model.func(model_path="model.joblib", tool_call_id="test-id")

        assert mock_dump.call_count == 2

    @patch("src.core.tools.deployment.joblib.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_summary_contains_artifact_paths(self, mock_file, mock_dump) -> None:
        """Verifies the summary lists all three artifact paths."""
        _populate_store_with_trained_model()

        result = deploy_model.func(model_path="model.joblib", tool_call_id="test-id")

        summary = result.update["messages"][0].content
        assert "model.joblib" in summary
        assert "model_scaler.joblib" in summary
        assert "model_metadata.json" in summary
        assert "DEPLOYMENT COMPLETE" in summary

    @patch("src.core.tools.deployment.joblib.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_scaler_path_derived_from_model_path(self, mock_file, mock_dump) -> None:
        """Verifies the scaler file path is derived from the model path."""
        _populate_store_with_trained_model()

        deploy_model.func(model_path="outputs/my_model.joblib", tool_call_id="test-id")

        dumped_paths = [call.args[1] for call in mock_dump.call_args_list]
        assert "outputs/my_model_scaler.joblib" in dumped_paths
