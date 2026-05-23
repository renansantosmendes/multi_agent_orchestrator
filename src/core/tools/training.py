import inspect
from typing import Annotated, Any, Dict, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.core.tools.store import pipeline_store


@tool
def train_model(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Trains an ML model on the already preprocessed fetal_health dataset.

    Args:
        model_name: Model name — 'RandomForest', 'LogisticRegression',
                    or 'GradientBoosting'.
        params: Optional hyperparameters (e.g. {'n_estimators': 200}).
    """
    if "X_train" not in pipeline_store:
        msg = "❌ Error: run preprocess_data before training."
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
            }
        )

    X_train = pipeline_store["X_train"]
    X_test = pipeline_store["X_test"]
    y_train = pipeline_store["y_train"]
    y_test = pipeline_store["y_test"]

    params = params or {}

    model_map = {
        "randomforest": (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
        "logisticregression": (LogisticRegression, {"max_iter": 2000}),
        "gradientboosting": (GradientBoostingClassifier, {"n_estimators": 100, "random_state": 42}),
    }

    key = model_name.lower().replace(" ", "").replace("_", "")
    model_class, defaults = model_map.get(key, (RandomForestClassifier, {"n_estimators": 100}))

    merged = {**defaults, **params}
    valid_args = set(inspect.signature(model_class).parameters.keys())
    filtered = {k: v for k, v in merged.items() if k in valid_args}

    model = model_class(**filtered)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    pipeline_store["trained_model"] = model
    pipeline_store["y_pred"] = y_pred
    pipeline_store["last_model_name"] = model_name
    pipeline_store["last_accuracy"] = acc

    history = pipeline_store.get("models_trained", [])
    history.append({"name": model_name, "accuracy": round(acc, 4), "params": filtered})
    pipeline_store["models_trained"] = history

    summary = (
        f"🏋️ TRAINING COMPLETE\n"
        f"   Model: {model_name}\n"
        f"   Params: {filtered}\n"
        f"   Accuracy: {acc:.4f} ({acc:.1%})"
    )

    return Command(
        update={
            "accuracy_history": [round(acc, 4)],
            "trained_models": [model_name],
            "best_model_name": model_name,
            "best_accuracy": round(acc, 4),
            "current_stage": "training",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )
