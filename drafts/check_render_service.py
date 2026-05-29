import time
import requests


RENDER_API_KEY = "rnd_TXWfE6ef5Qrxsex8bLHEG1rBvSvI"
SERVICE_ID = "srv-d8948re7r5hc73evu4ag"

BASE_URL = f"https://api.render.com/v1/services/{SERVICE_ID}"
headers = {"Authorization": f"Bearer {RENDER_API_KEY}", "Content-Type": "application/json"}


def get_service_info() -> dict:
    """Fetch current service metadata from Render."""
    response = requests.get(BASE_URL, headers=headers, verify=False)
    response.raise_for_status()
    return response.json()


def trigger_deploy() -> str:
    """Trigger a new deploy and return the deploy ID."""
    response = requests.post(f"{BASE_URL}/deploys", headers=headers, verify=False)
    response.raise_for_status()
    deploy_id = response.json()["id"]
    print(f"Deploy triggered: {deploy_id}")
    return deploy_id


def wait_for_deploy(deploy_id: str, max_attempts: int = 20, interval: int = 30) -> None:
    """Poll the deploy status until it succeeds, fails, or times out."""
    for attempt in range(1, max_attempts + 1):
        response = requests.get(
            f"{BASE_URL}/deploys/{deploy_id}",
            headers=headers,
            verify=False,
        )
        response.raise_for_status()
        status = response.json().get("status")
        print(f"Attempt {attempt}/{max_attempts} — status: {status}")

        if status == "live":
            print("Deploy succeeded!")
            return
        if "failed" in status or status == "canceled":
            raise RuntimeError(f"Deploy failed with status: {status}")

        time.sleep(interval)

    raise TimeoutError(f"Deploy timed out after {max_attempts * interval // 60} minutes")


if __name__ == "__main__":
    info = get_service_info()
    print(f"Service: {info['name']} — status: {info['suspended']}")

    # deploy_id = trigger_deploy()
    # wait_for_deploy(deploy_id)
