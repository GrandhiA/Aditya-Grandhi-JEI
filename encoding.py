import pandas as pd

from utils import (
    print_header,
    print_subheader
)

def detect_categorical_columns(X):
    categorical = list(
        X.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    return categorical


def detect_numeric_columns(X):
    numeric = list(
        X.select_dtypes(
            include=["number"]
        ).columns
    )

    return numeric


def print_encoding_summary(X):

    print_header("ENCODING SUMMARY")

    categorical = detect_categorical_columns(X)

    numeric = detect_numeric_columns(X)

    print(f"Numeric Features     : {len(numeric)}")
    print(f"Categorical Features : {len(categorical)}")

    if len(categorical) > 0:

        print_subheader("Categorical Columns")

        for col in categorical:

            print(col)


def encode_categorical_features(
        X,
        drop_first=True
):
    print_header("ONE-HOT ENCODING")

    categorical = detect_categorical_columns(X)

    if len(categorical) == 0:

        print("No categorical variables found.")

        return X.copy()

    print("Encoding columns:")

    for col in categorical:

        print(f"  {col}")

    X_encoded = pd.get_dummies(
        X,
        columns=categorical,
        drop_first=drop_first,
        dtype=float
    )

    print(
        f"\nFeatures before encoding : {X.shape[1]}"
    )

    print(
        f"Features after encoding  : {X_encoded.shape[1]}"
    )

    return X_encoded


def align_train_test_features(
        X_train,
        X_test
):
    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0
    )

    return X_train, X_test


def verify_numeric_features(X):
    """
    Confirm every feature is numeric.
    """

    bad_columns = []

    for col in X.columns:

        if not pd.api.types.is_numeric_dtype(
                X[col]
        ):

            bad_columns.append(col)

    if len(bad_columns) == 0:

        print("\nAll features are numeric.")

        return

    print("\nNon-numeric columns detected:")

    for col in bad_columns:

        print(col)

    raise ValueError(
        "Encoding failed. Dataset still contains "
        "non-numeric columns."
    )


def feature_name_report(X):

    print_header("ENCODED FEATURE LIST")

    print(f"Total Features: {X.shape[1]}")

    print()

    for feature in X.columns:

        print(feature)


def encoding_pipeline(
        X,
        drop_first=True,
        print_features=False
):
    print_encoding_summary(X)

    X = encode_categorical_features(
        X,
        drop_first=drop_first
    )

    verify_numeric_features(X)

    if print_features:

        feature_name_report(X)

    return X