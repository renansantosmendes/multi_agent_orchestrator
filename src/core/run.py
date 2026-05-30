import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.agents.orchestrator import run_pipeline
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def main() -> None:
    """Executes the full ML pipeline using the multi-agent orchestrator."""
    run_pipeline()


if __name__ == "__main__":
    main()
