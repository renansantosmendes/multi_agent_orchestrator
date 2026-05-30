import json
from typing import Annotated

import joblib
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from src.core.logging_config import get_logger
from src.core.tools.store import pipeline_store

logger = get_logger(__name__)


@tool
def deploy_model(
    model_path: str = "best_model.joblib",
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Serializes the trained model with joblib, saves the scaler,
    and generates a JSON metadata file.

    Args:
        model_path: File path of the .joblib file to save the model.
    """
    logger.info("Deployment started | model_path=%s", model_path)

    if "trained_model" not in pipeline_store:
        logger.error("Deployment aborted | reason=no trained model in store")
        msg = "❌ Error: no model available for deployment. Run train_model first."
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
        )

    model = pipeline_store["trained_model"]
    scaler = pipeline_store.get("scaler")
    model_name = pipeline_store.get("last_model_name", "unknown")
    accuracy = pipeline_store.get("last_accuracy", 0.0)
    feature_cols = pipeline_store.get("feature_cols", [])

    joblib.dump(model, model_path)
    logger.info("Model serialized | path=%s", model_path)

    scaler_path = model_path.replace(".joblib", "_scaler.joblib")
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler serialized | path=%s", scaler_path)

    metadata = {
        "model_name": model_name,
        "accuracy": accuracy,
        "features": feature_cols,
        "model_file": model_path,
        "scaler_file": scaler_path,
    }
    meta_path = model_path.replace(".joblib", "_metadata.json")

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved | path=%s accuracy=%.4f", meta_path, accuracy)

    logger.info(
        "Deployment complete | model=%s accuracy=%.4f artifacts=[%s, %s, %s]",
        model_name, accuracy, model_path, scaler_path, meta_path,
    )

    summary = (
        f"DEPLOYMENT COMPLETE\n"
        f"   Model:    {model_name} (accuracy: {accuracy:.4f})\n"
        f"   Artifacts:\n"
        f"     - Model:    {model_path}\n"
        f"     - Scaler:   {scaler_path}\n"
        f"     - Metadata: {meta_path}\n"
        f"   Status: Ready for production"
    )

    return Command(
        update={
            "model_path": model_path,
            "current_stage": "deployed",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )
