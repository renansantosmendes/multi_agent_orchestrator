from typing import Annotated

import mlflow
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from src.core.logging_config import get_logger
from src.core.tools.mlflow_config import configure_dagshub_tracking
from src.core.tools.store import pipeline_store

logger = get_logger(__name__)


@tool
def register_model(
    registered_model_name: str = "fetal_health_classifier",
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Registers the best trained model in the MLflow Model Registry.

    Reuses the run_id generated during training to register the artifact
    already logged — no re-upload of the model artifact.

    Args:
        registered_model_name: Name used when registering the model in the MLflow registry.
    """
    logger.info("Model registration started | registry_name=%s", registered_model_name)

    run_id = pipeline_store.get("last_run_id")
    artifact_uri = pipeline_store.get("last_model_artifact_uri")
    model_name = pipeline_store.get("last_model_name", "unknown")
    accuracy = pipeline_store.get("last_accuracy", 0.0)

    if not run_id:
        logger.error("Registration aborted | reason=no run_id found in pipeline store")
        msg = "Error: no MLflow run_id found. Run train_model before registering."
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
        )

    model_uri = artifact_uri if artifact_uri else f"runs:/{run_id}/model"
    logger.info("Registering model | model_uri=%s model=%s accuracy=%.4f", model_uri, model_name, accuracy)
    configure_dagshub_tracking()

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    version = model_version.version
    pipeline_store["mlflow_run_id"] = run_id
    pipeline_store["model_version"] = version

    logger.info(
        "Registration complete | registry=%s version=%s run_id=%s",
        registered_model_name, version, run_id,
    )

    summary = (
        f"MODEL REGISTRATION COMPLETE\n"
        f"   Model type:    {model_name} (accuracy: {accuracy:.4f})\n"
        f"   Run ID:        {run_id}\n"
        f"   Registry name: {registered_model_name}\n"
        f"   Version:       {version}\n"
        f"   Status: Ready"
    )

    return Command(
        update={
            "mlflow_run_id": run_id,
            "model_version": str(version),
            "current_stage": "registered",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )
