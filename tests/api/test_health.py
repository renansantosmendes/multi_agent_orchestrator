from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_returns_200(self) -> None:
        """Verifies the endpoint responds with HTTP 200."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_status_is_ok(self) -> None:
        """Verifies the response body contains status 'ok'."""
        response = client.get("/health")

        assert response.json()["status"] == "ok"

    def test_version_is_present(self) -> None:
        """Verifies the response body contains a non-empty version field."""
        response = client.get("/health")
        data = response.json()

        assert "version" in data
        assert len(data["version"]) > 0

    def test_response_matches_schema(self) -> None:
        """Verifies the response contains exactly the expected schema keys."""
        response = client.get("/health")
        data = response.json()

        assert set(data.keys()) == {"status", "version"}
