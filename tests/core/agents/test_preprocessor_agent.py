from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import END
from sklearn.preprocessing import StandardScaler

from src.core.agents.preprocessor_agent import PreprocessorAgent, PreprocessorState


def make_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
        "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0, 10.0],
        "target": [1, 2, 3, 1, 2, 1],
    })


def make_mock_openai_client(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


class TestPreprocessorAgentInit:
    def test_compiles_graph(self) -> None:
        agent = PreprocessorAgent()
        assert agent.graph is not None


class TestProfilingNode:
    def test_report_contains_all_keys(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._profiling_node(state)
        report = command.update["profiling_report"]
        for key in (
            "shape", "dtypes", "null_counts", "total_nulls",
            "descriptive_stats", "class_distribution", "skewness",
            "high_correlation_pairs", "duplicate_rows",
        ):
            assert key in report

    def test_shape_matches_input(self) -> None:
        dataframe = make_sample_dataframe()
        state: PreprocessorState = {"raw_dataframe": dataframe, "target_column": "target"}
        command = PreprocessorAgent._profiling_node(state)
        assert command.update["profiling_report"]["shape"] == dataframe.shape

    def test_detects_duplicate_rows(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._profiling_node(state)
        assert command.update["profiling_report"]["duplicate_rows"] == 1

    def test_detects_high_correlation(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._profiling_node(state)
        pairs = command.update["profiling_report"]["high_correlation_pairs"]
        assert len(pairs) > 0

    def test_null_counts_are_zero_for_clean_data(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._profiling_node(state)
        assert command.update["profiling_report"]["total_nulls"] == 0

    def test_class_distribution_contains_target_values(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._profiling_node(state)
        distribution = command.update["profiling_report"]["class_distribution"]
        assert set(distribution.keys()) == {1, 2, 3}

    def test_routes_to_llm_insight_node(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._profiling_node(state)
        assert command.goto == "llm_insight_node"


class TestLlmInsightNode:
    def test_returns_text_from_response(self) -> None:
        state: PreprocessorState = {"profiling_report": {"shape": (100, 5)}}
        mock_client = make_mock_openai_client("Dataset looks balanced.")
        with patch("openai.OpenAI", return_value=mock_client):
            command = PreprocessorAgent._llm_insight_node(state)
        assert command.update["llm_insight"] == "Dataset looks balanced."

    def test_handles_none_content_gracefully(self) -> None:
        state: PreprocessorState = {"profiling_report": {}}
        mock_client = make_mock_openai_client("")
        mock_client.chat.completions.create.return_value.choices[0].message.content = None
        with patch("openai.OpenAI", return_value=mock_client):
            command = PreprocessorAgent._llm_insight_node(state)
        assert command.update["llm_insight"] == ""

    def test_handles_api_failure_gracefully(self) -> None:
        state: PreprocessorState = {"profiling_report": {}}
        with patch("openai.OpenAI", side_effect=Exception("Timeout")):
            command = PreprocessorAgent._llm_insight_node(state)
        assert "LLM call failed" in command.update["llm_insight"]
        assert "Timeout" in command.update["llm_insight"]

    def test_routes_to_preprocessing_node(self) -> None:
        state: PreprocessorState = {"profiling_report": {}}
        mock_client = make_mock_openai_client("insight")
        with patch("openai.OpenAI", return_value=mock_client):
            command = PreprocessorAgent._llm_insight_node(state)
        assert command.goto == "preprocessing_node"


class TestPreprocessingNode:
    def test_drops_duplicate_rows(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._preprocessing_node(state)
        assert len(command.update["processed_dataframe"]) == 5

    def test_feature_names_excludes_target(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._preprocessing_node(state)
        assert "target" not in command.update["feature_names"]

    def test_scaler_is_standard_scaler(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._preprocessing_node(state)
        assert isinstance(command.update["scaler"], StandardScaler)

    def test_sanitizes_column_names_with_spaces(self) -> None:
        dataframe = pd.DataFrame({
            "Feature A": [1.0, 2.0, 3.0],
            "Feature B": [4.0, 5.0, 6.0],
            "Target Col": [1, 2, 3],
        })
        state: PreprocessorState = {"raw_dataframe": dataframe, "target_column": "Target Col"}
        command = PreprocessorAgent._preprocessing_node(state)
        assert "feature_a" in command.update["feature_names"]
        assert "feature_b" in command.update["feature_names"]

    def test_target_column_updated_to_sanitized_name(self) -> None:
        dataframe = pd.DataFrame({
            "Feature A": [1.0, 2.0, 3.0],
            "Target Col": [1, 2, 3],
        })
        state: PreprocessorState = {"raw_dataframe": dataframe, "target_column": "Target Col"}
        command = PreprocessorAgent._preprocessing_node(state)
        assert command.update["target_column"] == "target_col"

    def test_routes_to_splitting_node(self) -> None:
        state: PreprocessorState = {
            "raw_dataframe": make_sample_dataframe(),
            "target_column": "target",
        }
        command = PreprocessorAgent._preprocessing_node(state)
        assert command.goto == "splitting_node"


class TestSplittingNode:
    def _make_processed_dataframe(self, n_samples: int = 100) -> pd.DataFrame:
        return pd.DataFrame({
            "feature_a": np.random.randn(n_samples),
            "feature_b": np.random.randn(n_samples),
            "target": np.tile([1, 2], n_samples // 2),
        })

    def test_train_test_split_sizes(self) -> None:
        state: PreprocessorState = {
            "processed_dataframe": self._make_processed_dataframe(100),
            "target_column": "target",
            "test_size": 0.2,
            "random_state": 42,
        }
        command = PreprocessorAgent._splitting_node(state)
        assert command.update["train_input_data"].shape[0] == 80
        assert command.update["test_input_data"].shape[0] == 20

    def test_output_arrays_match_split_sizes(self) -> None:
        state: PreprocessorState = {
            "processed_dataframe": self._make_processed_dataframe(100),
            "target_column": "target",
            "test_size": 0.2,
            "random_state": 42,
        }
        command = PreprocessorAgent._splitting_node(state)
        assert command.update["train_output_data"].shape[0] == 80
        assert command.update["test_output_data"].shape[0] == 20

    def test_uses_default_test_size_when_not_set(self) -> None:
        state: PreprocessorState = {
            "processed_dataframe": self._make_processed_dataframe(100),
            "target_column": "target",
        }
        command = PreprocessorAgent._splitting_node(state)
        assert command.update["train_input_data"].shape[0] == 80

    def test_feature_columns_excluded_from_output(self) -> None:
        state: PreprocessorState = {
            "processed_dataframe": self._make_processed_dataframe(100),
            "target_column": "target",
            "test_size": 0.2,
            "random_state": 42,
        }
        command = PreprocessorAgent._splitting_node(state)
        assert command.update["train_input_data"].shape[1] == 2

    def test_routes_to_end(self) -> None:
        state: PreprocessorState = {
            "processed_dataframe": self._make_processed_dataframe(100),
            "target_column": "target",
            "test_size": 0.2,
            "random_state": 42,
        }
        command = PreprocessorAgent._splitting_node(state)
        assert command.goto == END


class TestPreprocessorAgentRun:
    def test_run_returns_train_test_arrays(self) -> None:
        dataframe = pd.DataFrame({
            "feature_a": np.random.randn(60),
            "feature_b": np.random.randn(60),
            "target": np.tile([1.0, 2.0, 3.0], 20),
        })
        agent = PreprocessorAgent()
        mock_client = make_mock_openai_client("Good dataset.")
        with patch("openai.OpenAI", return_value=mock_client):
            result = agent.run(dataframe, "target")
        assert "train_input_data" in result
        assert "test_input_data" in result
        assert "train_output_data" in result
        assert "test_output_data" in result

    def test_run_returns_llm_insight(self) -> None:
        dataframe = pd.DataFrame({
            "feature_a": np.random.randn(60),
            "target": np.tile([1.0, 2.0], 30),
        })
        agent = PreprocessorAgent()
        mock_client = make_mock_openai_client("Insight text here.")
        with patch("openai.OpenAI", return_value=mock_client):
            result = agent.run(dataframe, "target")
        assert result["llm_insight"] == "Insight text here."

    def test_run_returns_profiling_report(self) -> None:
        dataframe = pd.DataFrame({
            "feature_a": np.random.randn(60),
            "target": np.tile([1.0, 2.0], 30),
        })
        agent = PreprocessorAgent()
        mock_client = make_mock_openai_client("ok")
        with patch("openai.OpenAI", return_value=mock_client):
            result = agent.run(dataframe, "target")
        assert "profiling_report" in result
