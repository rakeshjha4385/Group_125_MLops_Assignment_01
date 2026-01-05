import os
import requests

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
OUT_PATH = "data/processed.cleveland.data"


def download_data(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(out_path, "wb") as file:
        file.write(response.content)

    print(f"Downloaded: {out_path}")


if __name__ == "__main__":
    download_data(UCI_URL, OUT_PATH)
