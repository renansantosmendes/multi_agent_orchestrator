from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from agents.trainer_agent import TrainerAgent


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    agent = TrainerAgent(model=RandomForestClassifier(n_estimators=100, random_state=42))
    result = agent.run(X_train, y_train, X_test, y_test)

    print("Metrics:", result["metrics"])


if __name__ == "__main__":
    main()
