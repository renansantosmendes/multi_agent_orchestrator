import inspect
from typing import Annotated, Any, Dict, Optional
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")

import mlflow
import mlflow.sklearn
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.core.logging_config import get_logger
from src.core.tools.mlflow_config import configure_dagshub_tracking
from src.core.tools.store import pipeline_store

logger = get_logger(__name__)


@tool
def train_model(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Trains an ML model on the already preprocessed fetal_health dataset.

    Uses MLflow autolog to capture hyperparameters, metrics, and model
    artifacts automatically. The experiment name is read from the pipeline
    store so all models in the same run share the same MLflow experiment.

    Args:
        model_name: Model name — 'RandomForest', 'LogisticRegression',
                    or 'GradientBoosting'.
        params: Optional hyperparameters (e.g. {'n_estimators': 200}).
    """
    experiment_name = pipeline_store.get("experiment_name", "agentic_ml_pipeline")
    logger.info("Training started | model=%s experiment=%s", model_name, experiment_name)

    if "X_train" not in pipeline_store:
        logger.error("Training aborted | reason=preprocessing not run")
        msg = "❌ Error: run preprocess_data before training."
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
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
    logger.info("Model instantiated | class=%s params=%s", model_class.__name__, filtered)

    configure_dagshub_tracking()
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(
        log_models=False,
        log_input_examples=False,
        log_post_training_metrics=False,
        silent=True,
    )

    model = model_class(**filtered)

    run_name = f"fetal_health_{model_name}_{uuid4().hex[:8]}"
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_param("feature_count", len(pipeline_store.get("feature_cols", [])))
        mlflow.log_metric("test_accuracy", acc)
        run_id = run.info.run_id
        model_artifact_uri = mlflow.get_artifact_uri("model")

    logger.info("Model trained | accuracy=%.4f run_id=%s", acc, run_id)

    pipeline_store["trained_model"] = model
    pipeline_store["y_pred"] = y_pred
    pipeline_store["last_model_name"] = model_name
    pipeline_store["last_accuracy"] = acc
    pipeline_store["last_run_id"] = run_id
    pipeline_store["last_model_artifact_uri"] = model_artifact_uri

    run_ids = pipeline_store.get("mlflow_run_ids", {})
    run_ids[model_name] = run_id
    pipeline_store["mlflow_run_ids"] = run_ids

    history = pipeline_store.get("models_trained", [])
    history.append({"name": model_name, "accuracy": round(acc, 4), "params": filtered, "run_id": run_id})
    pipeline_store["models_trained"] = history

    summary = (
        f"TRAINING COMPLETE\n"
        f"   Model:    {model_name}\n"
        f"   Params:   {filtered}\n"
        f"   Accuracy: {acc:.4f} ({acc:.1%})\n"
        f"   MLflow run ID: {run_id}"
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
