from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestMain:
    """Tests for the root main.py entry point."""

    @patch("src.core.agents.orchestrator.run_pipeline")
    def test_run_pipeline_is_importable(self, mock_run_pipeline: MagicMock) -> None:
        """Verifies run_pipeline can be imported from the orchestrator module."""
        from src.core.agents.orchestrator import run_pipeline

        assert callable(run_pipeline)
