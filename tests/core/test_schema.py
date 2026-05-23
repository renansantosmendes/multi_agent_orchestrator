from __future__ import annotations

from src.core.schema import MLPipelineContext


class TestMLPipelineContext:
    """Tests for the MLPipelineContext TypedDict schema."""

    def test_has_message_fields(self) -> None:
        """Verifies that message-related fields are declared."""
        keys = MLPipelineContext.__annotations__
        assert "messages" in keys

    def test_has_drift_fields(self) -> None:
        """Verifies that all drift-related fields are declared."""
        keys = MLPipelineContext.__annotations__
        assert "drift_detected" in keys
        assert "drift_summary" in keys
        assert "drifted_columns" in keys

    def test_has_preprocessing_fields(self) -> None:
        """Verifies that all preprocessing-related fields are declared."""
        keys = MLPipelineContext.__annotations__
        assert "preprocessing_done" in keys
        assert "feature_columns" in keys
        assert "target_column" in keys

    def test_has_training_fields(self) -> None:
        """Verifies that all training-related fields are declared."""
        keys = MLPipelineContext.__annotations__
        assert "accuracy_history" in keys
        assert "trained_models" in keys
        assert "best_model_name" in keys
        assert "best_accuracy" in keys

    def test_has_deploy_fields(self) -> None:
        """Verifies that all deployment-related fields are declared."""
        keys = MLPipelineContext.__annotations__
        assert "model_path" in keys
        assert "current_stage" in keys
