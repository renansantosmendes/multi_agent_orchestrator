from typing import Annotated

import mlflow
import mlflow.sklearn
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from src.core.tools.store import pipeline_store


@tool
def register_model(
    experiment_name: str = "ml_pipeline",
    registered_model_name: str = "fetal_health_classifier",
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Registers the best trained model in MLflow, logging metrics and artifacts.

    Args:
        experiment_name: MLflow experiment name to log the run under.
        registered_model_name: Name used when registering the model in the MLflow registry.
    """
    if "trained_model" not in pipeline_store:
        msg = "Error: no trained model found. Run train_model before registering."
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
        )

    model = pipeline_store["trained_model"]
    model_name = pipeline_store.get("last_model_name", "unknown")
    accuracy = pipeline_store.get("last_accuracy", 0.0)
    feature_cols = pipeline_store.get("feature_cols", [])

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_metric("accuracy", accuracy)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

    run_id = run.info.run_id
    model_version = model_info.registered_model_version

    pipeline_store["mlflow_run_id"] = run_id
    pipeline_store["model_version"] = model_version

    summary = (
        f"MODEL REGISTRATION COMPLETE\n"
        f"   Experiment:    {experiment_name}\n"
        f"   Model type:    {model_name} (accuracy: {accuracy:.4f})\n"
        f"   Run ID:        {run_id}\n"
        f"   Registry name: {registered_model_name}\n"
        f"   Version:       {model_version}\n"
        f"   Status: Ready"
    )

    return Command(
        update={
            "mlflow_run_id": run_id,
            "model_version": str(model_version),
            "current_stage": "registered",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )
