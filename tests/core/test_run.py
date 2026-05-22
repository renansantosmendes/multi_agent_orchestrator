from __future__ import annotations

import pytest

from src.core.run import main


class TestMain:
    def test_runs_without_error(self, capsys: pytest.CaptureFixture) -> None:
        main()
        captured = capsys.readouterr()
        assert "Metrics:" in captured.out

    def test_outputs_classification_metrics(self, capsys: pytest.CaptureFixture) -> None:
        main()
        captured = capsys.readouterr()
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in captured.out
