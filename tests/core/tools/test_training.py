from __future__ import annotations

import numpy as np
import pytest
from langgraph.types import Command
from sklearn.datasets import make_classification

from src.core.tools import store
from src.core.tools.training import train_model


def _populate_store_with_data() -> None:
    """Fills the pipeline_store with a small synthetic classification dataset."""
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3,
                                n_informative=3, random_state=42)
    y = y.astype(float) + 1.0
    split = 80
    store.pipeline_store["X_train"] = X[:split]
    store.pipeline_store["X_test"] = X[split:]
    store.pipeline_store["y_train"] = y[:split]
    store.pipeline_store["y_test"] = y[split:]


@pytest.fixture(autouse=True)
def reset_store() -> None:
    """Clears the pipeline_store before and after every test."""
    store.pipeline_store.clear()
    yield
    store.pipeline_store.clear()


class TestTrainModel:
    """Tests for the train_model tool."""

    def test_error_when_store_not_populated(self) -> None:
        """Verifies an error message is returned when preprocessing was not run first."""
        result = train_model.func(model_name="RandomForest", tool_call_id="test-id")

        assert isinstance(result, Command)
        assert "Error" in result.update["messages"][0].content

    def test_trains_random_forest(self) -> None:
        """Verifies a RandomForest is trained and stored successfully."""
        _populate_store_with_data()

        result = train_model.func(model_name="RandomForest", tool_call_id="test-id")

        assert isinstance(result, Command)
        assert "trained_model" in store.pipeline_store
        assert result.update["current_stage"] == "training"

    def test_trains_logistic_regression(self) -> None:
        """Verifies a LogisticRegression is trained and stored successfully."""
        _populate_store_with_data()

        result = train_model.func(model_name="LogisticRegression", tool_call_id="test-id")

        assert isinstance(result, Command)
        assert "trained_model" in store.pipeline_store

    def test_trains_gradient_boosting(self) -> None:
        """Verifies a GradientBoosting model is trained and stored successfully."""
        _populate_store_with_data()

        result = train_model.func(model_name="GradientBoosting", tool_call_id="test-id")

        assert isinstance(result, Command)
        assert "trained_model" in store.pipeline_store

    def test_accuracy_stored_in_command_update(self) -> None:
        """Verifies the accuracy is stored in the Command update and pipeline_store."""
        _populate_store_with_data()

        result = train_model.func(model_name="RandomForest", tool_call_id="test-id")

        assert len(result.update["accuracy_history"]) == 1
        assert 0.0 <= result.update["accuracy_history"][0] <= 1.0
        assert store.pipeline_store["last_accuracy"] == result.update["best_accuracy"]

    def test_history_accumulates_across_calls(self) -> None:
        """Verifies the models_trained history grows with each call."""
        _populate_store_with_data()

        train_model.func(model_name="RandomForest", tool_call_id="test-id-1")
        train_model.func(model_name="LogisticRegression", tool_call_id="test-id-2")

        assert len(store.pipeline_store["models_trained"]) == 2
