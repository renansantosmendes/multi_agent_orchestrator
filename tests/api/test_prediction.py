from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)

VALID_PAYLOAD = {"features": [0.1, 0.5, 0.3, 0.8, 1.2, 0.9, 0.4, 0.6, 0.2, 0.7]}


class TestPredictionEndpoint:
    """Tests for the POST /prediction endpoint."""

    def test_returns_200_with_valid_payload(self) -> None:
        """Verifies the endpoint responds with HTTP 200 for a valid request."""
        response = client.post("/prediction", json=VALID_PAYLOAD)

        assert response.status_code == 200

    def test_prediction_is_valid_class(self) -> None:
        """Verifies the prediction value is one of the three fetal health classes."""
        response = client.post("/prediction", json=VALID_PAYLOAD)

        assert response.json()["prediction"] in (1, 2, 3)

    def test_label_is_valid(self) -> None:
        """Verifies the label corresponds to a known fetal health class name."""
        response = client.post("/prediction", json=VALID_PAYLOAD)

        assert response.json()["label"] in ("Normal", "Suspect", "Pathological")

    def test_confidence_is_between_0_and_1(self) -> None:
        """Verifies the confidence score is a valid probability."""
        response = client.post("/prediction", json=VALID_PAYLOAD)

        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_status_is_mock(self) -> None:
        """Verifies the status field flags this as a mock response."""
        response = client.post("/prediction", json=VALID_PAYLOAD)

        assert response.json()["status"] == "mock"

    def test_response_matches_schema(self) -> None:
        """Verifies the response contains exactly the expected schema keys."""
        response = client.post("/prediction", json=VALID_PAYLOAD)
        data = response.json()

        assert set(data.keys()) == {"prediction", "label", "confidence", "status"}

    def test_returns_422_when_features_missing(self) -> None:
        """Verifies the endpoint rejects a request with no features field."""
        response = client.post("/prediction", json={})

        assert response.status_code == 422

    def test_returns_422_when_payload_is_empty(self) -> None:
        """Verifies the endpoint rejects a completely empty body."""
        response = client.post("/prediction", json=None)

        assert response.status_code == 422

    def test_accepts_single_feature(self) -> None:
        """Verifies the endpoint accepts a feature list with a single element."""
        response = client.post("/prediction", json={"features": [0.5]})

        assert response.status_code == 200

    def test_accepts_empty_feature_list(self) -> None:
        """Verifies the endpoint accepts an empty feature list."""
        response = client.post("/prediction", json={"features": []})

        assert response.status_code == 200
