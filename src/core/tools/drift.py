import warnings
from typing import Annotated

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from src.core.logging_config import get_logger
from src.core.tools.store import DATASET_URL

logger = get_logger(__name__)


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
    logger.info(
        "Drift detection started | reference=[%d:%d] current=[%d:%d] threshold=%.0f%%",
        reference_start, reference_end, current_start, current_end,
        drift_share_threshold * 100,
    )

    df = pd.read_csv(DATASET_URL)
    reference_data = df.iloc[reference_start:reference_end]
    current_data = df.iloc[current_start:current_end]
    logger.info("Dataset loaded | total_rows=%d", len(df))

    report = Report(metrics=[DataDriftPreset()], include_tests="True")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
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
            logger.debug("Column drift detected | column=%s p_value=%.4f", col_name, p_value)
        column_details.append(
            f"  - {col_name}: p-value={p_value:.4f} | "
            f"drift={'YES' if has_drift else 'NO'}"
        )

    drift_detected = drift_share >= drift_share_threshold

    if drift_detected:
        logger.warning(
            "Global drift detected | drifted=%d/%d (%.1f%%) threshold=%.0f%%",
            int(drift_count), len(metrics) - 1, drift_share * 100,
            drift_share_threshold * 100,
        )
    else:
        logger.info(
            "No global drift | drifted=%d/%d (%.1f%%)",
            int(drift_count), len(metrics) - 1, drift_share * 100,
        )

    summary = "\n".join([
        "DATA DRIFT REPORT",
        f"   Reference: rows [{reference_start}:{reference_end}] "
        f"({reference_end - reference_start} samples)",
        f"   Current:   rows [{current_start}:{current_end}] "
        f"({current_end - current_start} samples)",
        "",
        f"   Drifted columns: {int(drift_count)} / {len(metrics)-1} "
        f"({drift_share:.1%})",
        f"   Global threshold: {drift_share_threshold:.0%}",
        f"   GLOBAL DRIFT: {'YES' if drift_detected else 'NO'}",
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
