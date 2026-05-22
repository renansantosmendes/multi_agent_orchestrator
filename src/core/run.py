from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.core.agents.trainer_agent import TrainerAgent


def main() -> None:
    input_data, output_data = load_iris(return_X_y=True)
    train_input, test_input, train_output, test_output = train_test_split(
        input_data, output_data, test_size=0.2, random_state=42
    )

    agent = TrainerAgent(model=RandomForestClassifier(n_estimators=100, random_state=42))
    result = agent.run(train_input, train_output, test_input, test_output)

    print("Metrics:", result["metrics"])


if __name__ == "__main__":
    main()
