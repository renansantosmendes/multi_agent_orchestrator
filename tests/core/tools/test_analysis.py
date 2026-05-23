from __future__ import annotations

import numpy as np
import pytest
from langgraph.types import Command
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.core.tools import store
from src.core.tools.analysis import analyze_results


def _populate_store_with_rf_model() -> None:
    """Trains a small RandomForest and loads all required keys into pipeline_store."""
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3,
                                n_informative=3, random_state=42)
    y = y.astype(float) + 1.0
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    store.pipeline_store.update({
        "trained_model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "last_model_name": "RandomForest",
        "feature_cols": [f"feature_{i}" for i in range(5)],
        "models_trained": [{"name": "RandomForest", "accuracy": 0.85}],
    })


def _populate_store_with_lr_model() -> None:
    """Trains a small LogisticRegression and loads all required keys into pipeline_store."""
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3,
                                n_informative=3, random_state=42)
    y = y.astype(float) + 1.0
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    store.pipeline_store.update({
        "trained_model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "last_model_name": "LogisticRegression",
        "feature_cols": [f"feature_{i}" for i in range(5)],
        "models_trained": [{"name": "LogisticRegression", "accuracy": 0.80}],
    })


@pytest.fixture(autouse=True)
def reset_store() -> None:
    """Clears the pipeline_store before and after every test."""
    store.pipeline_store.clear()
    yield
    store.pipeline_store.clear()


class TestAnalyzeResults:
    """Tests for the analyze_results tool."""

    def test_error_when_no_model_in_store(self) -> None:
        """Verifies an error message is returned when no trained model exists."""
        result = analyze_results.func(tool_call_id="test-id")

        assert isinstance(result, Command)
        assert "Error" in result.update["messages"][0].content

    def test_returns_command_with_trained_model(self) -> None:
        """Verifies a Command is returned when a trained model is in the store."""
        _populate_store_with_rf_model()

        result = analyze_results.func(tool_call_id="test-id")

        assert isinstance(result, Command)
        assert result.update["current_stage"] == "evaluation"

    def test_summary_contains_classification_report(self) -> None:
        """Verifies the summary includes the classification report."""
        _populate_store_with_rf_model()

        result = analyze_results.func(tool_call_id="test-id")

        summary = result.update["messages"][0].content
        assert "Classification Report" in summary

    def test_summary_includes_feature_importances_for_rf(self) -> None:
        """Verifies the summary lists feature importances for a tree-based model."""
        _populate_store_with_rf_model()

        result = analyze_results.func(tool_call_id="test-id")

        summary = result.update["messages"][0].content
        assert "TOP 10 FEATURES" in summary

    def test_summary_includes_coef_for_logistic_regression(self) -> None:
        """Verifies the summary uses coefficient magnitude for linear models."""
        _populate_store_with_lr_model()

        result = analyze_results.func(tool_call_id="test-id")

        summary = result.update["messages"][0].content
        assert "TOP 10 FEATURES" in summary

    def test_model_comparison_shown_when_multiple_models(self) -> None:
        """Verifies that a comparison section appears when more than one model was trained."""
        _populate_store_with_rf_model()
        store.pipeline_store["models_trained"] = [
            {"name": "RandomForest", "accuracy": 0.85},
            {"name": "LogisticRegression", "accuracy": 0.80},
        ]

        result = analyze_results.func(tool_call_id="test-id")

        summary = result.update["messages"][0].content
        assert "MODEL COMPARISON" in summary
        assert "Best model" in summary
