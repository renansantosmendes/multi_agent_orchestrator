import json
from typing import Annotated

import joblib
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from src.core.tools.store import pipeline_store


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
    if "trained_model" not in pipeline_store:
        msg = "❌ Error: no model available for deployment. Run train_model first."
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
            }
        )

    model = pipeline_store["trained_model"]
    scaler = pipeline_store.get("scaler")
    model_name = pipeline_store.get("last_model_name", "unknown")
    accuracy = pipeline_store.get("last_accuracy", 0.0)
    feature_cols = pipeline_store.get("feature_cols", [])

    joblib.dump(model, model_path)

    scaler_path = model_path.replace(".joblib", "_scaler.joblib")
    if scaler is not None:
        joblib.dump(scaler, scaler_path)

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

    summary = (
        f"🚀 DEPLOYMENT COMPLETE\n"
        f"   Model:    {model_name} (accuracy: {accuracy:.4f})\n"
        f"   Artifacts:\n"
        f"     - Model:    {model_path}\n"
        f"     - Scaler:   {scaler_path}\n"
        f"     - Metadata: {meta_path}\n"
        f"   Status: ✅ Ready for production"
    )

    return Command(
        update={
            "model_path": model_path,
            "current_stage": "deployed",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )
