import operator
from typing import Annotated, List

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class MLPipelineContext(TypedDict):
    """Shared state schema for the ML pipeline LangGraph graph."""

    messages: Annotated[List[BaseMessage], operator.add]
    drift_detected: bool
    drift_summary: str
    drifted_columns: Annotated[List[str], operator.add]
    preprocessing_done: bool
    feature_columns: Annotated[List[str], operator.add]
    target_column: str
    accuracy_history: Annotated[List[float], operator.add]
    trained_models: Annotated[List[str], operator.add]
    best_model_name: str
    best_accuracy: float
    experiment_name: str
    model_path: str
    mlflow_run_id: str
    model_version: str
    current_stage: str
