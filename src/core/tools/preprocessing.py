from typing import Annotated

import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.core.tools.store import DATASET_URL, pipeline_store


@tool
def preprocess_data(
    target_column: str = "fetal_health",
    test_size: float = 0.2,
    random_state: int = 42,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Preprocesses the fetal_health dataset: separates features and target,
    normalizes with StandardScaler, and performs a stratified train/test split.

    Args:
        target_column: Name of the target column.
        test_size: Fraction of data to use for testing.
        random_state: Reproducibility seed.
    """
    df = pd.read_csv(DATASET_URL)

    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols
    )

    pipeline_store["X_train"] = X_train_scaled
    pipeline_store["X_test"] = X_test_scaled
    pipeline_store["y_train"] = y_train.values
    pipeline_store["y_test"] = y_test.values
    pipeline_store["scaler"] = scaler
    pipeline_store["feature_cols"] = feature_cols

    summary = (
        f"⚙️ PREPROCESSING COMPLETE\n"
        f"   Features: {len(feature_cols)} columns\n"
        f"   Train: {X_train_scaled.shape[0]} samples\n"
        f"   Test:  {X_test_scaled.shape[0]} samples\n"
        f"   Scaler: StandardScaler\n"
        f"   Stratify: Yes"
    )

    return Command(
        update={
            "preprocessing_done": True,
            "feature_columns": feature_cols,
            "target_column": target_column,
            "current_stage": "preprocessing",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )
