import json
import inspect
import operator
import warnings
from typing import Annotated, List, Dict, Any

import numpy as np
import pandas as pd
import joblib
from typing_extensions import TypedDict

# LangChain / LangGraph / DeepAgents
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Evidently
from evidently import Report
from evidently.presets import DataDriftPreset

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")
print("✅ Imports OK")


# %% CELL 4 — Constants and Global Store
DATASET_URL = (
    "https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023"
    "/master/fetal_health.csv"
)

# Global store — shared memory across subagent tools
_pipeline_store: Dict[str, Any] = {}

print("✅ Constants defined")


# %% CELL 5 — Context Schema (shared graph state)
class MLPipelineContext(TypedDict):
    """Shared state schema for the ML pipeline graph."""

    messages: Annotated[List[BaseMessage], operator.add]

    # Drift
    drift_detected: bool
    drift_summary: str
    drifted_columns: Annotated[List[str], operator.add]

    # Preprocessing
    preprocessing_done: bool
    feature_columns: Annotated[List[str], operator.add]
    target_column: str

    # Training
    accuracy_history: Annotated[List[float], operator.add]
    trained_models: Annotated[List[str], operator.add]
    best_model_name: str
    best_accuracy: float

    # Deploy
    model_path: str
    current_stage: str

print("✅ MLPipelineContext defined")


# %% CELL 6 — Tool: detect_data_drift (DriftAgent)
@tool
def detect_data_drift(
    reference_start: int,
    reference_end: int,
    current_start: int,
    current_end: int,
    drift_share_threshold: float = 0.5,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Detects data drift between a reference slice and a current slice
    of the fetal_health dataset using Evidently DataDriftPreset.

    Args:
        reference_start: Start index of the reference data.
        reference_end: End index of the reference data.
        current_start: Start index of the current data.
        current_end: End index of the current data.
        drift_share_threshold: Fraction of drifted columns required to
                               trigger global drift (default 0.5).
    """
    df = pd.read_csv(DATASET_URL)
    reference_data = df.iloc[reference_start:reference_end]
    current_data = df.iloc[current_start:current_end]

    report = Report(metrics=[DataDriftPreset()], include_tests="True")
    result = report.run(current_data=current_data, reference_data=reference_data)
    result_dict = result.dict()

    metrics = result_dict.get("metrics", [])
    drift_count_metric = metrics[0] if metrics else {}
    drift_count = drift_count_metric.get("value", {}).get("count", 0)
    drift_share = drift_count_metric.get("value", {}).get("share", 0.0)

    drifted_cols = []
    column_details = []
    for m in metrics[1:]:
        col_name = m.get("config", {}).get("column", "unknown")
        p_value = m.get("value", 1.0)
        threshold = m.get("config", {}).get("threshold", 0.05)
        has_drift = p_value < threshold
        if has_drift:
            drifted_cols.append(col_name)
        column_details.append(
            f"  - {col_name}: p-value={p_value:.4f} | "
            f"drift={'YES' if has_drift else 'NO'}"
        )

    drift_detected = drift_share >= drift_share_threshold

    summary = "\n".join([
        "📊 DATA DRIFT REPORT",
        f"   Reference: rows [{reference_start}:{reference_end}] "
        f"({reference_end - reference_start} samples)",
        f"   Current:   rows [{current_start}:{current_end}] "
        f"({current_end - current_start} samples)",
        "",
        f"   Drifted columns: {int(drift_count)} / {len(metrics)-1} "
        f"({drift_share:.1%})",
        f"   Global threshold: {drift_share_threshold:.0%}",
        f"   🚨 GLOBAL DRIFT: {'YES' if drift_detected else 'NO'}",
        "",
        "   Details per column:",
    ] + column_details)

    return Command(
        update={
            "drift_detected": drift_detected,
            "drift_summary": summary,
            "drifted_columns": drifted_cols,
            "current_stage": "drift_check",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )

print("✅ Tool: detect_data_drift")


# %% CELL 7 — Tool: preprocess_data (PreprocessAgent)
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

    _pipeline_store["X_train"] = X_train_scaled
    _pipeline_store["X_test"] = X_test_scaled
    _pipeline_store["y_train"] = y_train.values
    _pipeline_store["y_test"] = y_test.values
    _pipeline_store["scaler"] = scaler
    _pipeline_store["feature_cols"] = feature_cols

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

print("✅ Tool: preprocess_data")


@tool
def train_model(
    model_name: str,
    params: Dict[str, Any] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Trains an ML model on the already preprocessed fetal_health dataset.

    Args:
        model_name: Model name — 'RandomForest', 'LogisticRegression',
                    or 'GradientBoosting'.
        params: Optional hyperparameters (e.g. {'n_estimators': 200}).
    """
    if "X_train" not in _pipeline_store:
        msg = "❌ Error: run preprocess_data before training."
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
            }
        )

    X_train = _pipeline_store["X_train"]
    X_test = _pipeline_store["X_test"]
    y_train = _pipeline_store["y_train"]
    y_test = _pipeline_store["y_test"]

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

    _pipeline_store["trained_model"] = model
    _pipeline_store["y_pred"] = y_pred
    _pipeline_store["last_model_name"] = model_name
    _pipeline_store["last_accuracy"] = acc

    history = _pipeline_store.get("models_trained", [])
    history.append({"name": model_name, "accuracy": round(acc, 4), "params": filtered})
    _pipeline_store["models_trained"] = history

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

print("✅ Tool: train_model")


@tool
def analyze_results(
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Generates a detailed report with classification_report, feature importances,
    and a comparison across all models trained in the session."""

    if "trained_model" not in _pipeline_store:
        msg = "❌ Error: no trained model found. Run train_model first."
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
            }
        )

    y_test = _pipeline_store["y_test"]
    y_pred = _pipeline_store["y_pred"]
    model = _pipeline_store["trained_model"]
    model_name = _pipeline_store.get("last_model_name", "unknown")
    feature_cols = _pipeline_store.get("feature_cols", [])
    models_history = _pipeline_store.get("models_trained", [])

    report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Suspect", "Pathological"],
    )

    importance_text = ""
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        top = importances[:10]
        lines = [f"  {i+1}. {n}: {v:.4f}" for i, (n, v) in enumerate(top)]
        importance_text = "\n\n   🏆 TOP 10 FEATURES:\n" + "\n".join(lines)
    elif hasattr(model, "coef_"):
        avg_coef = np.abs(model.coef_).mean(axis=0)
        importances = sorted(
            zip(feature_cols, avg_coef),
            key=lambda x: x[1], reverse=True,
        )
        top = importances[:10]
        lines = [f"  {i+1}. {n}: {v:.4f}" for i, (n, v) in enumerate(top)]
        importance_text = (
            "\n\n   🏆 TOP 10 FEATURES (mean |coef|):\n" + "\n".join(lines)
        )

    comparison_text = ""
    if len(models_history) > 1:
        sorted_models = sorted(models_history, key=lambda x: x["accuracy"], reverse=True)
        comp_lines = [
            f"  {'→' if m['name'] == model_name else ' '} "
            f"{m['name']}: {m['accuracy']:.4f}"
            for m in sorted_models
        ]
        comparison_text = (
            "\n\n   📊 MODEL COMPARISON:\n" + "\n".join(comp_lines)
        )
        best = sorted_models[0]
        comparison_text += (
            f"\n\n   ✅ Best model: {best['name']} ({best['accuracy']:.4f})"
        )

    summary = (
        f"📈 RESULTS ANALYSIS — {model_name}\n\n"
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

print("✅ Tool: analyze_results")


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
    if "trained_model" not in _pipeline_store:
        msg = "❌ Error: no model available for deployment. Run train_model first."
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
            }
        )

    model = _pipeline_store["trained_model"]
    scaler = _pipeline_store.get("scaler")
    model_name = _pipeline_store.get("last_model_name", "unknown")
    accuracy = _pipeline_store.get("last_accuracy", 0.0)
    feature_cols = _pipeline_store.get("feature_cols", []) 

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

print("✅ Tool: deploy_model")


drift_agent = {
    "name": "drift_detector",
    "description": (
        "Specialist in data drift detection. Delegate to it when you need "
        "to check whether the data distribution has changed. "
        "It uses Evidently to compare reference data vs current data."
    ),
    "system_prompt": (
        "You are the DataDriftDetectorAgent, a specialist in drift detection.\n\n"
        "YOUR RESPONSIBILITIES:\n"
        "- Use the detect_data_drift tool to analyse drift in the data\n"
        "- Interpret the results: which columns drifted, p-values\n"
        "- Issue a clear verdict: drift detected or not\n\n"
        "When called, use detect_data_drift with the provided parameters "
        "and return a structured summary of the result. "
        "NEVER include courtesy phrases, farewells, or offers of further help. "
        "Be direct and technical."
    ),
    "tools": [detect_data_drift],
}

preprocess_agent = {
    "name": "preprocessor",
    "description": (
        "Specialist in data preprocessing. Delegate to it to "
        "normalise features, separate the target, and perform a train/test split."
    ),
    "system_prompt": (
        "You are the PreprocessAgent, a specialist in data preparation.\n\n"
        "YOUR RESPONSIBILITIES:\n"
        "- Use the preprocess_data tool to prepare the dataset\n"
        "- Ensure the data is normalised and ready for training\n"
        "- Report: number of features, train/test sizes\n\n"
        "When called, run preprocess_data and return the summary. "
        "NEVER include courtesy phrases, farewells, or offers of further help. "
        "Be direct and technical."
    ),
    "tools": [preprocess_data],
}

trainer_agent = {
    "name": "trainer",
    "description": (
        "Specialist in ML model training. Delegate to it to train models "
        "such as RandomForest, LogisticRegression, or GradientBoosting. "
        "It can train multiple models in sequence to compare results."
    ),
    "system_prompt": (
        "You are the TrainerAgent, a specialist in model training.\n\n"
        "YOUR RESPONSIBILITIES:\n"
        "- Use the train_model tool to train models\n"
        "- You can train MULTIPLE models if requested\n"
        "- Available models: RandomForest, LogisticRegression, GradientBoosting\n"
        "- Report the accuracy of each trained model\n\n"
        "If the user asks to find the best model, train at least "
        "2 different models and compare the results.\n"
        "Always return a summary with the accuracy of all trained models. "
        "NEVER include courtesy phrases, farewells, or offers of further help. "
        "Be direct and technical."
    ),
    "tools": [train_model],
}

analyzer_agent = {
    "name": "result_analyzer",
    "description": (
        "Specialist in ML results analysis. Delegate to it to generate "
        "classification reports, feature importances, and model comparisons."
    ),
    "system_prompt": (
        "You are the ResultAnalyzerAgent, a specialist in model evaluation.\n\n"
        "YOUR RESPONSIBILITIES:\n"
        "- Use the analyze_results tool to generate detailed reports\n"
        "- Interpret precision, recall, and f1-score per class\n"
        "- Highlight the most important features\n"
        "- Compare models if more than one was trained\n"
        "- Provide a final recommendation on which model to use\n\n"
        "Return a clear and actionable analysis. "
        "NEVER include courtesy phrases, farewells, or offers of further help. "
        "Be direct and technical."
    ),
    "tools": [analyze_results],
}

deploy_agent = {
    "name": "deployer",
    "description": (
        "Specialist in model deployment. Delegate to it to serialise "
        "the trained model, save the scaler, and write metadata."
    ),
    "system_prompt": (
        "You are the DeployAgent, a specialist in model deployment.\n\n"
        "YOUR RESPONSIBILITIES:\n"
        "- Use the deploy_model tool to save the model for production\n"
        "- Verify that the model, scaler, and metadata were saved\n"
        "- Report the paths of the generated artifacts\n\n"
        "Run the deployment and return the final status. "
        "NEVER include courtesy phrases, farewells, or offers of further help. "
        "Be direct and technical."
    ),
    "tools": [deploy_model],
}

print("✅ SubAgents defined: drift_detector, preprocessor, trainer, result_analyzer, deployer")


ORCHESTRATOR_PROMPT = """You are the Main Orchestrator of an ML pipeline for the fetal_health dataset.
You do NOT execute tasks directly. You DELEGATE to specialised subagents using the `task` tool.

AVAILABLE SUBAGENTS:
- drift_detector: checks for data drift
- preprocessor: preprocesses data
- trainer: trains models
- result_analyzer: analyses results
- deployer: deploys the model

MANDATORY FLOW:
1. Delegate to `drift_detector` to check for drift (reference_start=0, reference_end=200, current_start=1000, current_end=1200)
2. Delegate to `preprocessor` to prepare the data
3. Delegate to `trainer` to train the requested model(s)
4. Delegate to `result_analyzer` to evaluate the results
5. Delegate to `deployer` to deploy the model

RULES:
- Execute ALL steps in sequence without interruption.
- NEVER pause to ask the user whether to proceed. You are autonomous.
- If drift is detected, log a WARNING and continue the pipeline.
- Briefly explain your reasoning BEFORE each delegation.
- At the end, present a consolidated summary of the entire pipeline.
"""


def create_ml_orchestrator(model_name: str = "gpt-4o-mini", temperature: float = 0):
    """Creates the orchestrator with a subagent architecture."""
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    orchestrator = create_deep_agent(
        model=llm,
        tools=[],
        subagents=[
            drift_agent,
            preprocess_agent,
            trainer_agent,
            analyzer_agent,
            deploy_agent,
        ],
        context_schema=MLPipelineContext,
        system_prompt=ORCHESTRATOR_PROMPT,
    )
    return orchestrator

print("✅ Orchestrator with SubAgents configured")


orchestrator = create_ml_orchestrator()

initial_state = {
    "messages": [
        HumanMessage(
            content=(
                "Run the full ML pipeline. "
                "Check for drift, preprocess the data, "
                "train a RandomForest and a LogisticRegression, "
                "analyse the results, and deploy the best model."
            )
        )
    ],
    "drift_detected": False,
    "drift_summary": "",
    "drifted_columns": [],
    "preprocessing_done": False,
    "feature_columns": [],
    "target_column": "",
    "accuracy_history": [],
    "trained_models": [],
    "best_model_name": "",
    "best_accuracy": 0.0,
    "model_path": "",
    "current_stage": "start",
}

print("🚀 Starting ML Pipeline with SubAgents...\n")

try:
    for event in orchestrator.stream(initial_state, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        agent_name = tc.get("args", {}).get("subagent_type", tc.get("name", "?"))
                        print(f"🛠️  Delegating to: {agent_name}")
                elif isinstance(last_msg, ToolMessage):
                    print(f"📦 Result:\n{last_msg.content}\n")
                else:
                    print(f"🤖 Orchestrator:\n{last_msg.content}\n")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
