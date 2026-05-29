from unittest.mock import MagicMock, patch

import pytest
from sklearn.ensemble import RandomForestClassifier

from src.core.tools.store import pipeline_store


@pytest.fixture(autouse=True)
def clean_pipeline_store():
    """Reset pipeline_store state before each test."""
    pipeline_store.clear()
    yield
    pipeline_store.clear()


@pytest.fixture()
def trained_model_in_store():
    """Populate pipeline_store with a minimal trained model."""
    model = RandomForestClassifier(n_estimators=2, random_state=0)
    model.fit([[0, 1], [1, 0]], [0, 1])
    pipeline_store["trained_model"] = model
    pipeline_store["last_model_name"] = "RandomForest"
    pipeline_store["last_accuracy"] = 0.95
    pipeline_store["feature_cols"] = ["f1", "f2"]


def _make_mock_run(run_id: str = "abc123"):
    """Build a mock MLflow ActiveRun with the given run_id."""
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    return mock_run


def _make_mock_model_info(version: str = "1"):
    """Build a mock MLflow ModelInfo with a registered_model_version."""
    mock_info = MagicMock()
    mock_info.registered_model_version = version
    return mock_info


@patch("src.core.tools.registration.mlflow.start_run")
@patch("src.core.tools.registration.mlflow.sklearn.log_model")
@patch("src.core.tools.registration.mlflow.log_metric")
@patch("src.core.tools.registration.mlflow.log_param")
@patch("src.core.tools.registration.mlflow.set_experiment")
def test_register_model_success(
    mock_set_experiment,
    mock_log_param,
    mock_log_metric,
    mock_log_model,
    mock_start_run,
    trained_model_in_store,
):
    """register_model returns a Command with run_id and model_version on success."""
    from src.core.tools.registration import register_model

    mock_start_run.return_value.__enter__ = lambda _: _make_mock_run("run-001")
    mock_start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_log_model.return_value = _make_mock_model_info("2")

    result = register_model.func(
        experiment_name="test_exp",
        registered_model_name="test_model",
        tool_call_id="tc1",
    )

    assert result.update["mlflow_run_id"] == "run-001"
    assert result.update["model_version"] == "2"
    assert result.update["current_stage"] == "registered"
    mock_set_experiment.assert_called_once_with("test_exp")
    mock_log_metric.assert_called_once_with("accuracy", 0.95)


def test_register_model_without_trained_model():
    """register_model returns an error message when no model is in the store."""
    from src.core.tools.registration import register_model

    result = register_model.func(
        experiment_name="test_exp",
        registered_model_name="test_model",
        tool_call_id="tc2",
    )

    message_content = result.update["messages"][0].content
    assert "no trained model found" in message_content.lower()


@patch("src.core.tools.registration.mlflow.start_run")
@patch("src.core.tools.registration.mlflow.sklearn.log_model")
@patch("src.core.tools.registration.mlflow.log_metric")
@patch("src.core.tools.registration.mlflow.log_param")
@patch("src.core.tools.registration.mlflow.set_experiment")
def test_register_model_logs_params(
    mock_set_experiment,
    mock_log_param,
    mock_log_metric,
    mock_log_model,
    mock_start_run,
    trained_model_in_store,
):
    """register_model logs model_type and feature_count as MLflow params."""
    from src.core.tools.registration import register_model

    mock_start_run.return_value.__enter__ = lambda _: _make_mock_run("run-002")
    mock_start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_log_model.return_value = _make_mock_model_info("3")

    register_model.func(
        experiment_name="exp",
        registered_model_name="model",
        tool_call_id="tc3",
    )

    calls = {call.args[0]: call.args[1] for call in mock_log_param.call_args_list}
    assert calls["model_type"] == "RandomForest"
    assert calls["feature_count"] == 2
