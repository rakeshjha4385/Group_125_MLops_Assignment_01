import requests

# URL for processed Cleveland heart disease dataset (change as needed)
uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
out_path = "data/processed.cleveland.data"

response = requests.get(uci_url)

with open(out_path, "wb") as f:
    f.write(response.content)

print("Downloaded:", out_path)
