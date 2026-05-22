from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing_extensions import TypedDict

load_dotenv()


class PreprocessorState(TypedDict, total=False):
    """Shared state flowing through the preprocessing agent."""

    raw_dataframe: pd.DataFrame
    target_column: str
    test_size: float
    random_state: int
    profiling_report: dict[str, Any]
    llm_insight: str
    processed_dataframe: pd.DataFrame
    train_input_data: np.ndarray
    train_output_data: np.ndarray
    test_input_data: np.ndarray
    test_output_data: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


class PreprocessorAgent:
    """A LangGraph-based agent that profiles, analyses, preprocesses, and
    splits a tabular dataset, preparing it for model training.

    The graph flows through four sequential nodes:

    1. **profiling_node** — generates a statistical profile of the raw data.
    2. **llm_insight_node** — sends the profile to an LLM and retrieves
       domain-specific insights about the dataset.
    3. **preprocessing_node** — cleans, encodes and scales the features.
    4. **splitting_node** — performs train/test split and outputs arrays
       ready for ``TrainerAgent``.

    Attributes:
        graph: The compiled LangGraph ``StateGraph`` ready for invocation.
    """

    def __init__(self) -> None:
        """Initialise the agent and compile the preprocessing graph."""
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph state graph.

        Returns:
            A compiled ``StateGraph`` with profiling, LLM insight,
            preprocessing, and splitting nodes.
        """
        builder = StateGraph(PreprocessorState)

        builder.add_node("profiling_node", self._profiling_node)
        builder.add_node("llm_insight_node", self._llm_insight_node)
        builder.add_node("preprocessing_node", self._preprocessing_node)
        builder.add_node("splitting_node", self._splitting_node)

        builder.add_edge(START, "profiling_node")

        return builder.compile()

    @staticmethod
    def _profiling_node(
        state: PreprocessorState,
    ) -> Command[Literal["llm_insight_node"]]:
        """Generate a statistical profile of the raw dataframe.

        Collects shape, dtypes, null counts, descriptive statistics, class
        balance, skewness and correlation highlights.

        Args:
            state: Must contain ``raw_dataframe`` and ``target_column``.

        Returns:
            A ``Command`` that stores the profiling report and routes to the
            LLM insight node.
        """
        dataframe = state["raw_dataframe"]
        target_column = state["target_column"]

        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        correlation_matrix = dataframe[numeric_columns].corr()
        high_correlation_pairs = []
        for i, col_a in enumerate(numeric_columns):
            for col_b in numeric_columns[i + 1 :]:
                value = correlation_matrix.loc[col_a, col_b]
                if abs(value) >= 0.7:
                    high_correlation_pairs.append((col_a, col_b, round(value, 3)))

        report: dict[str, Any] = {
            "shape": dataframe.shape,
            "dtypes": dataframe.dtypes.astype(str).to_dict(),
            "null_counts": dataframe.isnull().sum().to_dict(),
            "total_nulls": int(dataframe.isnull().sum().sum()),
            "descriptive_stats": dataframe.describe().round(3).to_dict(),
            "class_distribution": (
                dataframe[target_column].value_counts().to_dict()
                if target_column in dataframe.columns
                else {}
            ),
            "skewness": dataframe[numeric_columns].skew().round(3).to_dict(),
            "high_correlation_pairs": high_correlation_pairs,
            "duplicate_rows": int(dataframe.duplicated().sum()),
        }

        return Command(update={"profiling_report": report}, goto="llm_insight_node")

    @staticmethod
    def _llm_insight_node(
        state: PreprocessorState,
    ) -> Command[Literal["preprocessing_node"]]:
        """Send the profiling report to an LLM and collect insights.

        Uses the OpenAI Chat Completions API to generate domain-specific
        observations, potential quality issues, and feature engineering
        suggestions based on the profiling report.

        Args:
            state: Must contain ``profiling_report``.

        Returns:
            A ``Command`` that stores the ``llm_insight`` string and routes
            to the preprocessing node.
        """
        import json

        report = state["profiling_report"]

        prompt = (
            "You are a senior data scientist. "
            "Below is the statistical profile of a dataset about fetal health "
            "classification from cardiotocography exams.\n\n"
            f"```json\n{json.dumps(report, indent=2, default=str)}\n```\n\n"
            "Based on this profile, provide a concise analysis covering:\n"
            "1. Class imbalance assessment and recommended handling strategy.\n"
            "2. Features with high skewness or unusual distributions.\n"
            "3. Highly correlated feature pairs and whether any should be dropped.\n"
            "4. Any data quality concerns (duplicates, outliers, nulls).\n"
            "5. Two concrete feature engineering ideas for this domain.\n\n"
            "Be specific and reference actual column names and numbers."
        )

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            insight_text = response.choices[0].message.content or ""
        except Exception as error:
            insight_text = f"LLM call failed: {error}"

        return Command(
            update={"llm_insight": insight_text}, goto="preprocessing_node"
        )

    @staticmethod
    def _preprocessing_node(
        state: PreprocessorState,
    ) -> Command[Literal["splitting_node"]]:
        """Clean, encode and scale the dataframe.

        Performs the following steps in order: drop duplicate rows, sanitise
        column names, separate features from target, cast the target to
        integer labels, and apply ``StandardScaler`` to numeric features.

        Args:
            state: Must contain ``raw_dataframe`` and ``target_column``.

        Returns:
            A ``Command`` that stores the processed dataframe, scaler, and
            feature names, then routes to the splitting node.
        """
        dataframe = state["raw_dataframe"].copy()
        target_column = state["target_column"]

        dataframe = dataframe.drop_duplicates()

        dataframe.columns = (
            dataframe.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^\w]", "", regex=True)
        )
        clean_target = (
            target_column.strip().lower().replace(" ", "_")
        )
        clean_target = "".join(c for c in clean_target if c.isalnum() or c == "_")

        output_series = dataframe[clean_target].astype(int)
        feature_dataframe = dataframe.drop(columns=[clean_target])

        numeric_columns = feature_dataframe.select_dtypes(include="number").columns
        scaler = StandardScaler()
        feature_dataframe[numeric_columns] = scaler.fit_transform(
            feature_dataframe[numeric_columns]
        )

        processed = feature_dataframe.copy()
        processed[clean_target] = output_series

        return Command(
            update={
                "processed_dataframe": processed,
                "scaler": scaler,
                "feature_names": feature_dataframe.columns.tolist(),
                "target_column": clean_target,
            },
            goto="splitting_node",
        )

    @staticmethod
    def _splitting_node(
        state: PreprocessorState,
    ) -> Command[Literal["__end__"]]:
        """Split the processed data into train and test arrays.

        Args:
            state: Must contain ``processed_dataframe``, ``target_column``,
                ``test_size``, and ``random_state``.

        Returns:
            A ``Command`` with train/test arrays ready for a model and routes
            to ``END``.
        """
        dataframe = state["processed_dataframe"]
        target_column = state["target_column"]
        test_size = state.get("test_size", 0.2)
        random_state = state.get("random_state", 42)

        feature_matrix = dataframe.drop(columns=[target_column]).values
        target_vector = dataframe[target_column].values

        train_input, test_input, train_output, test_output = train_test_split(
            feature_matrix,
            target_vector,
            test_size=test_size,
            random_state=random_state,
            stratify=target_vector,
        )

        return Command(
            update={
                "train_input_data": train_input,
                "train_output_data": train_output,
                "test_input_data": test_input,
                "test_output_data": test_output,
            },
            goto=END,
        )

    def run(
        self,
        raw_dataframe: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Execute the full preprocessing pipeline.

        Args:
            raw_dataframe: The raw dataset as a pandas ``DataFrame``.
            target_column: Name of the target column.
            test_size: Fraction of data reserved for testing.
            random_state: Seed for reproducible splits.

        Returns:
            The final ``PreprocessorState`` dictionary with profiling report,
            LLM insight, and train/test arrays.
        """
        initial_state: PreprocessorState = {
            "raw_dataframe": raw_dataframe,
            "target_column": target_column,
            "test_size": test_size,
            "random_state": random_state,
        }

        return self.graph.invoke(initial_state)