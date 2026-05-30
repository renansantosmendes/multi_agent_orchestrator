import os

os.environ.setdefault("MPLBACKEND", "Agg")

import mlflow


def configure_dagshub_tracking() -> None:
    """Configures MLflow to use DagsHub as the remote tracking server.

    Reads DAGSHUB_URL, DAGSHUB_USER, and DAGSHUB_TOKEN from environment
    variables. Does nothing when any variable is missing (e.g. in tests).
    """
    dagshub_url = os.environ.get("DAGSHUB_URL", "").rstrip("/")
    user = os.environ.get("DAGSHUB_USER", "")
    token = os.environ.get("DAGSHUB_TOKEN", "")

    if not dagshub_url or not user or not token:
        return

    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(f"{dagshub_url}.mlflow")
