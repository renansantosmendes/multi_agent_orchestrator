from unittest.mock import MagicMock, patch

import pytest

from src.core.tools.store import pipeline_store


@pytest.fixture(autouse=True)
def clean_pipeline_store():
    """Reset pipeline_store state before each test."""
    pipeline_store.clear()
    yield
    pipeline_store.clear()


@pytest.fixture()
def trained_run_in_store():
    """Populate pipeline_store as if train_model already ran."""
    pipeline_store["last_run_id"] = "run-001"
    pipeline_store["last_model_artifact_uri"] = "https://dagshub.com/user/repo.mlflow/artifacts/run-001/model"
    pipeline_store["last_model_name"] = "RandomForest"
    pipeline_store["last_accuracy"] = 0.95


def _make_mock_model_version(version: str = "1"):
    """Build a mock MLflow ModelVersion with a version number."""
    mock_mv = MagicMock()
    mock_mv.version = version
    return mock_mv


@patch("src.core.tools.registration.mlflow.register_model")
@patch("src.core.tools.registration.configure_dagshub_tracking")
def test_register_model_success(mock_configure, mock_register, trained_run_in_store):
    """register_model returns a Command with run_id and model_version on success."""
    from src.core.tools.registration import register_model

    mock_register.return_value = _make_mock_model_version("2")

    result = register_model.func(
        tool_call_id="tc1",
    )

    assert result.update["mlflow_run_id"] == "run-001"
    assert result.update["model_version"] == "2"
    assert result.update["current_stage"] == "registered"
    mock_register.assert_called_once_with(
        model_uri="https://dagshub.com/user/repo.mlflow/artifacts/run-001/model",
        name="fetal_health",
    )


@patch("src.core.tools.registration.configure_dagshub_tracking")
def test_register_model_without_run_id(mock_configure):
    """register_model returns an error message when no run_id is in the store."""
    from src.core.tools.registration import register_model

    result = register_model.func(
        tool_call_id="tc2",
    )

    message_content = result.update["messages"][0].content
    assert "run_id" in message_content.lower()


@patch("src.core.tools.registration.mlflow.register_model")
@patch("src.core.tools.registration.configure_dagshub_tracking")
def test_register_model_uses_correct_uri(mock_configure, mock_register, trained_run_in_store):
    """register_model builds the model URI from the run_id stored in the pipeline."""
    from src.core.tools.registration import register_model

    mock_register.return_value = _make_mock_model_version("3")

    register_model.func(tool_call_id="tc3")

    uri_used = mock_register.call_args.kwargs["model_uri"]
    assert uri_used == "https://dagshub.com/user/repo.mlflow/artifacts/run-001/model"
