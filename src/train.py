import argparse
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.features import get_preprocessing_pipeline


def load_data(path):
    df = pd.read_csv(path, header=None)

    df.columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    df = df.replace(-9.0, np.nan)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    return df


def main(args):
    df = load_data(args.data)

    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    numerical_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]

    pipeline = get_preprocessing_pipeline(categorical_features, numerical_features)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    X_train_prep = pipeline.fit_transform(X_train)
    X_test_prep = pipeline.transform(X_test)

    if args.dry_run:
        print(
            "Dry run: Preprocessing and splits succeeded. Exiting before model training."
        )
        return

    # Train models (use full grid in production, smaller here for speed)
    logreg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(random_state=42)

    gs_logreg = GridSearchCV(
        logreg,
        {"C": [1]},
        cv=3,
    )
    gs_logreg.fit(X_train_prep, y_train)

    gs_rf = GridSearchCV(
        rf,
        {"n_estimators": [100]},
        cv=3,
    )
    gs_rf.fit(X_train_prep, y_train)

    os.makedirs("./models", exist_ok=True)

    joblib.dump(
        gs_logreg.best_estimator_,
        "./models/logreg.joblib",
    )

    joblib.dump(
        gs_rf.best_estimator_,
        "./models/random_forest.joblib",
    )

    joblib.dump(
        pipeline,
        "./models/preprocessing_pipeline.joblib",
    )

    print("Training completed. Models and pipeline saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data/processed.cleveland.data")

    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    main(args)
