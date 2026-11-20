# src/lab/data.py

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


class DataError(Exception):
    """Custom exception for data related issues."""
    pass


def load_data():
    """
    Load Online Shoppers dataset from OpenML.
    Returns:
        X (pd.DataFrame): features
        y (pd.Series): target (0/1)
    """

    try:
        data = fetch_openml(data_id=45560, as_frame=True)
    except Exception as e:
        raise DataError(f"Failed to download dataset: {e}")

    df = data.frame.copy()

    if "Revenue" not in df.columns:
        raise DataError("Target column 'Revenue' not found in dataset.")

    # -------- Target processing --------
    y = df["Revenue"].astype(str).str.lower().map({
        "true": 1,
        "false": 0
    })

    if y.isna().sum() > 0:
        raise DataError("Revenue conversion failed. Unexpected values found.")

    X = df.drop(columns=["Revenue"])

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Steps:
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
    """

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_cols = X.select_dtypes(
        exclude=["object", "category", "bool"]
    ).columns.tolist()

    if len(numeric_cols) == 0:
        raise DataError("No numeric columns detected.")

    if len(categorical_cols) == 0:
        raise DataError("No categorical columns detected.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                    ), categorical_cols),
        ]
    )

    return preprocessor