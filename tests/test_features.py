import numpy as np
import pandas as pd

from src.features import get_preprocessing_pipeline


def test_pipeline_handles_missing():
    cat = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    num = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    pipeline = get_preprocessing_pipeline(cat, num)

    df = pd.DataFrame(
        {
            "sex": [1, np.nan],
            "age": [63, np.nan],
            "cp": [1, 2],
            "trestbps": [145, 120],
            "chol": [233, 240],
            "fbs": [1, 0],
            "restecg": [2, 0],
            "thalach": [150, 160],
            "exang": [0, 1],
            "oldpeak": [2.3, np.nan],
            "slope": [3, 1],
            "ca": [0, 2],
            "thal": [6, 3],
        }
    )

    pipeline.fit_transform(df)
