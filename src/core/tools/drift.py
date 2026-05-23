from typing import Annotated

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from src.core.tools.store import DATASET_URL


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
