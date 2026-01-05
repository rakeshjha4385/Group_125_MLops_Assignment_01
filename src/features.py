from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def get_preprocessing_pipeline(categorical_features, numerical_features):

    # -----------------------------
    # Imputers for missing values
    # -----------------------------
    categorical_imputer = SimpleImputer(
        strategy="most_frequent"  # Fill NaN with most frequent value
    )

    numerical_imputer = SimpleImputer(strategy="mean")  # Fill NaN with mean

    # -----------------------------
    # Transformers
    # -----------------------------
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", categorical_imputer),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", numerical_imputer),
            ("scaler", StandardScaler()),
        ]
    )

    # -----------------------------
    # Column Transformer
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # -----------------------------
    # Final pipeline
    # -----------------------------
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline
