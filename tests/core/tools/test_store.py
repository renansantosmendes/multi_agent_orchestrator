from __future__ import annotations

from src.core.tools.store import DATASET_URL, pipeline_store


class TestStore:
    """Tests for the shared pipeline store and constants."""

    def test_dataset_url_is_string(self) -> None:
        """Verifies that DATASET_URL is a non-empty HTTPS string."""
        assert isinstance(DATASET_URL, str)
        assert DATASET_URL.startswith("https://")
        assert len(DATASET_URL) > 0

    def test_dataset_url_points_to_csv(self) -> None:
        """Verifies that DATASET_URL ends with a CSV filename."""
        assert DATASET_URL.endswith(".csv")

    def test_pipeline_store_is_dict(self) -> None:
        """Verifies that pipeline_store is a dictionary."""
        assert isinstance(pipeline_store, dict)
