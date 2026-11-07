import pandas as pd
from config import config

from utils import (
    print_header,
    print_subheader
)

from encoding import (
    encode_categorical_features
)

from feature_selection import (
    remove_constant_features,
    remove_highly_correlated_features
)

from scaling import (
    scale_train_test
)

def preprocess_features(
        X,
        config
):

    print_header("PREPROCESSING FEATURES")

    metadata = {}

    # --------------------------------------------------------
    # Remove constant columns
    # --------------------------------------------------------

    if config.REMOVE_CONSTANT_COLUMNS:

        X, removed = remove_constant_features(X)

        metadata["constant_removed"] = removed

    else:

        metadata["constant_removed"] = []

    # --------------------------------------------------------
    # Remove highly correlated columns
    # --------------------------------------------------------

    if config.CHECK_CORRELATION:

        X, removed = remove_highly_correlated_features(
            X,
            threshold=config.CORRELATION_THRESHOLD
        )

        metadata["correlated_removed"] = removed

    else:

        metadata["correlated_removed"] = []


    if config.ONE_HOT_ENCODE:

        X = encode_categorical_features(X)

    metadata["feature_names"] = list(X.columns)

    print("\nFinal number of features:", X.shape[1])

    return X, metadata


def preprocess_train_test(
        X_train,
        X_test,
        config
):

    print_header("TRAIN / TEST PREPROCESSING")

    scaler = None

    if config.SCALE_FEATURES:

        X_train, X_test, scaler = scale_train_test(
            X_train,
            X_test
        )

    return X_train, X_test, scaler


def feature_report(
        X,
        metadata
):

    print_header("FEATURE REPORT")

    print("Number of Features")

    print(len(X.columns))

    print_subheader("Removed Constant Features")

    if metadata["constant_removed"]:

        for feature in metadata["constant_removed"]:

            print(feature)

    else:

        print("None")

    print_subheader("Removed Correlated Features")

    if metadata["correlated_removed"]:

        for feature in metadata["correlated_removed"]:

            print(feature)

    else:

        print("None")

    print_subheader("Remaining Features")

    for feature in X.columns:

        print(feature)


def align_columns(
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


def verify_numeric_dataset(
        X
):

    print_header("VERIFYING NUMERIC FEATURES")

    bad_columns = []

    for column in X.columns:

        if not pd.api.types.is_numeric_dtype(X[column]):

            bad_columns.append(column)

    if len(bad_columns) == 0:

        print("All features are numeric.")

    else:

        print("Non-numeric columns detected:")

        for col in bad_columns:

            print(col)

        raise ValueError(
            "Dataset still contains non-numeric columns."
        )


def preprocessing_pipeline(
        X_train,
        X_test,
        config
):

    X_train, X_test = align_columns(
        X_train,
        X_test
    )

    verify_numeric_dataset(X_train)

    X_train, X_test, scaler = preprocess_train_test(
        X_train,
        X_test,
        config
    )

    return X_train, X_test, scaler