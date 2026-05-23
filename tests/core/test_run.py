from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.run import main


class TestMain:
    """Tests for the run.main entry point."""

    @patch("src.core.run.run_pipeline")
    def test_delegates_to_run_pipeline(self, mock_run_pipeline: MagicMock) -> None:
        """Verifies that main() delegates execution to run_pipeline."""
        main()

        mock_run_pipeline.assert_called_once()

    @patch("src.core.run.run_pipeline")
    def test_runs_without_error(self, mock_run_pipeline: MagicMock) -> None:
        """Verifies that main() completes without raising an exception."""
        mock_run_pipeline.return_value = None

        main()
