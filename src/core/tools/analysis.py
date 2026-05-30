from typing import Annotated

import numpy as np
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from sklearn.metrics import classification_report

from src.core.logging_config import get_logger
from src.core.tools.store import pipeline_store

logger = get_logger(__name__)


@tool
def analyze_results(
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Generates a detailed report with classification_report, feature importances,
    and a comparison across all models trained in the session."""
    logger.info("Results analysis started")

    if "trained_model" not in pipeline_store:
        logger.error("Analysis aborted | reason=no trained model in store")
        msg = "❌ Error: no trained model found. Run train_model first."
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
        )

    y_test = pipeline_store["y_test"]
    y_pred = pipeline_store["y_pred"]
    model = pipeline_store["trained_model"]
    model_name = pipeline_store.get("last_model_name", "unknown")
    feature_cols = pipeline_store.get("feature_cols", [])
    models_history = pipeline_store.get("models_trained", [])

    report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Suspect", "Pathological"],
    )
    logger.info("Classification report generated | model=%s", model_name)

    importance_text = ""
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        top = importances[:10]
        lines = [f"  {i+1}. {n}: {v:.4f}" for i, (n, v) in enumerate(top)]
        importance_text = "\n\n   TOP 10 FEATURES:\n" + "\n".join(lines)
        logger.info("Feature importances computed | top_feature=%s (%.4f)", top[0][0], top[0][1])
    elif hasattr(model, "coef_"):
        avg_coef = np.abs(model.coef_).mean(axis=0)
        importances = sorted(
            zip(feature_cols, avg_coef),
            key=lambda x: x[1], reverse=True,
        )
        top = importances[:10]
        lines = [f"  {i+1}. {n}: {v:.4f}" for i, (n, v) in enumerate(top)]
        importance_text = "\n\n   TOP 10 FEATURES (mean |coef|):\n" + "\n".join(lines)
        logger.info("Coefficient importances computed | top_feature=%s (%.4f)", top[0][0], top[0][1])

    comparison_text = ""
    if len(models_history) > 1:
        sorted_models = sorted(models_history, key=lambda x: x["accuracy"], reverse=True)
        comp_lines = [
            f"  {'>' if m['name'] == model_name else ' '} "
            f"{m['name']}: {m['accuracy']:.4f}"
            for m in sorted_models
        ]
        comparison_text = "\n\n   MODEL COMPARISON:\n" + "\n".join(comp_lines)
        best = sorted_models[0]
        comparison_text += f"\n\n   Best model: {best['name']} ({best['accuracy']:.4f})"
        logger.info(
            "Model comparison | best=%s accuracy=%.4f total_models=%d",
            best["name"], best["accuracy"], len(sorted_models),
        )

    logger.info("Results analysis complete | model=%s", model_name)

    summary = (
        f"RESULTS ANALYSIS — {model_name}\n\n"
        f"   Classification Report:\n{report}"
        f"{importance_text}"
        f"{comparison_text}"
    )

    return Command(
        update={
            "current_stage": "evaluation",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )
