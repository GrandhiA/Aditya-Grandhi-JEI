import pandas as pd
import numpy as np

from utils import (
    print_header,
    print_subheader
)

def missing_value_summary(df):
    """
    Print and return missing values for every column.
    """

    print_header("MISSING VALUE SUMMARY")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        print("No missing values detected.")

        return pd.Series(dtype=int)

    missing = missing.sort_values(
        ascending=False
    )

    summary = pd.DataFrame({

        "Feature": missing.index,
        "MissingValues": missing.values,
        "PercentMissing":
            (missing.values / len(df) * 100).round(2)

    })

    print(summary)

    return summary


def remove_constant_features(X):
    """
    Remove columns containing only one unique value.
    """

    print_header("REMOVE CONSTANT FEATURES")

    constant = []

    for column in X.columns:

        if X[column].nunique(dropna=False) <= 1:

            constant.append(column)

    if len(constant) == 0:

        print("No constant features found.")

        return X.copy(), []

    print("Removed constant features:")

    for feature in constant:

        print(f"  {feature}")

    X = X.drop(columns=constant)

    print(
        f"\nRemaining features: {X.shape[1]}"
    )

    return X, constant


def remove_duplicate_features(X):
    """
    Remove columns that are exact duplicates.
    """

    print_header("REMOVE DUPLICATE FEATURES")

    duplicates = []

    columns = list(X.columns)

    for i in range(len(columns)):

        col1 = columns[i]

        for j in range(i + 1, len(columns)):

            col2 = columns[j]

            if X[col1].equals(X[col2]):

                duplicates.append(col2)

    duplicates = sorted(list(set(duplicates)))

    if len(duplicates) == 0:

        print("No duplicate features found.")

        return X.copy(), []

    print("Removed duplicate features:")

    for feature in duplicates:

        print(f"  {feature}")

    X = X.drop(columns=duplicates)

    print(
        f"\nRemaining features: {X.shape[1]}"
    )

    return X, duplicates


def near_zero_variance_features(
        X,
        threshold=1e-4
):
    """
    Identify features with almost no variance.
    """

    print_header("LOW VARIANCE FEATURES")

    numeric = X.select_dtypes(
        include=["number"]
    )

    variances = numeric.var()

    low_variance = variances[
        variances < threshold
    ]

    if len(low_variance) == 0:

        print("No low-variance features found.")

        return []

    summary = pd.DataFrame({

        "Feature": low_variance.index,
        "Variance": low_variance.values

    })

    summary = summary.sort_values(
        by="Variance"
    )

    print(summary)

    return list(summary["Feature"])


def remove_low_variance_features(
        X,
        threshold=1e-4
):
    """
    Remove near-zero variance features.
    """

    print_header("REMOVE LOW VARIANCE FEATURES")

    low_variance = near_zero_variance_features(
        X,
        threshold
    )

    if len(low_variance) == 0:

        return X.copy(), []

    X = X.drop(columns=low_variance)

    print(
        f"\nRemaining features: {X.shape[1]}"
    )

    return X, low_variance


def dataset_quality_report(X):
    """
    Run all dataset quality checks without
    modifying the dataset.
    """

    print_header("DATASET QUALITY REPORT")

    missing_value_summary(X)

    print()

    constant = [
        c for c in X.columns
        if X[c].nunique(dropna=False) <= 1
    ]

    print_subheader("Constant Features")

    if len(constant) == 0:

        print("None")

    else:

        for feature in constant:

            print(feature)

    print()

    low = near_zero_variance_features(X)

    if len(low) == 0:

        print("\nNo low-variance features detected.")

def correlation_matrix(X):
    """
    Compute Pearson correlation matrix for numeric features.
    """

    print_header("CORRELATION MATRIX")

    numeric = X.select_dtypes(include=["number"])

    corr = numeric.corr(method="pearson")

    print(f"Computed {corr.shape[0]} x {corr.shape[1]} correlation matrix.")

    return corr


def save_correlation_matrix(
        corr,
        filename="correlation_matrix.csv"
):
    """
    Save correlation matrix to CSV.
    """

    corr.to_csv(filename)

    print(f"Saved correlation matrix to '{filename}'.")


def correlation_heatmap(
        corr,
        filename="correlation_heatmap.png",
        figsize=(14,12)
):
    """
    Save a correlation heatmap.
    """

    print_header("CORRELATION HEATMAP")

    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)

    plt.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    plt.colorbar(label="Pearson Correlation")

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns,
        fontsize=7
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved heatmap to '{filename}'.")


def find_highly_correlated_features(
        X,
        threshold=0.95
):
    """
    Find pairs of highly correlated features.

    Returns
    -------
    DataFrame
    """

    print_header("HIGHLY CORRELATED FEATURES")

    corr = correlation_matrix(X).abs()

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    pairs = []

    for column in upper.columns:

        for row in upper.index:

            value = upper.loc[row, column]

            if pd.notnull(value):

                if value >= threshold:

                    pairs.append({

                        "Feature_1": row,
                        "Feature_2": column,
                        "Correlation": value

                    })

    if len(pairs) == 0:

        print("No highly correlated features found.")

        return pd.DataFrame()

    pairs = pd.DataFrame(pairs)

    pairs = pairs.sort_values(
        by="Correlation",
        ascending=False
    )

    print(pairs)

    return pairs


def remove_highly_correlated_features(
        X,
        threshold=0.95
):
    """
    Remove one feature from each highly correlated pair.

    Returns
    -------
    X_filtered
    removed_features
    """

    print_header("REMOVE HIGHLY CORRELATED FEATURES")

    corr = correlation_matrix(X).abs()

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    remove = []

    for column in upper.columns:

        if any(upper[column] >= threshold):

            remove.append(column)

    remove = sorted(list(set(remove)))

    if len(remove) == 0:

        print("No features removed.")

        return X.copy(), []

    print("Removed Features:")

    for feature in remove:

        print(f"  {feature}")

    X = X.drop(columns=remove)

    print()

    print(f"Remaining Features: {X.shape[1]}")

    return X, remove


def correlation_report(
        X,
        threshold=0.95,
        save_csv=True
):
    """
    Complete correlation analysis.
    """

    print_header("CORRELATION REPORT")

    corr = correlation_matrix(X)

    if save_csv:

        save_correlation_matrix(corr)

    correlation_heatmap(corr)

    high_corr = find_highly_correlated_features(
        X,
        threshold
    )

    print()

    print(f"Highly correlated pairs: {len(high_corr)}")

    return corr, high_corr

def feature_selection_pipeline(
        X,
        correlation_threshold=0.95,
        remove_correlated=False,
        low_variance_threshold=1e-4,
        save_reports=True
):

    print_header("FEATURE SELECTION PIPELINE")

    X = X.copy()

    original_features = X.shape[1]

    missing_summary = missing_value_summary(X)

    X, constant_removed = remove_constant_features(X)

    X, duplicate_removed = remove_duplicate_features(X)

    X, variance_removed = remove_low_variance_features(
        X,
        threshold=low_variance_threshold
    )

    corr_matrix = correlation_matrix(X)

    high_corr = find_highly_correlated_features(
        X,
        threshold=correlation_threshold
    )

    correlated_removed = []

    if remove_correlated:

        X, correlated_removed = remove_highly_correlated_features(
            X,
            threshold=correlation_threshold
        )

    print_header("FEATURE SELECTION SUMMARY")

    print(f"Original Features : {original_features}")
    print(f"Final Features    : {X.shape[1]}")
    print()

    print(f"Constant Removed        : {len(constant_removed)}")
    print(f"Duplicate Removed       : {len(duplicate_removed)}")
    print(f"Low Variance Removed    : {len(variance_removed)}")
    print(f"Correlation Pairs Found : {len(high_corr)}")
    print(f"Correlated Removed      : {len(correlated_removed)}")

    summary = {

        "Original Features": original_features,

        "Final Features": X.shape[1],

        "Constant Removed": constant_removed,

        "Duplicate Removed": duplicate_removed,

        "Low Variance Removed": variance_removed,

        "Highly Correlated Pairs": high_corr,

        "Correlated Removed": correlated_removed

    }

    return X, summary


def save_feature_list(
        X,
        filename="selected_features.csv"
):
    """
    Save the remaining feature names.
    """

    feature_list = pd.DataFrame({

        "Feature": X.columns

    })

    feature_list.to_csv(
        filename,
        index=False
    )

    print(f"Saved feature list to '{filename}'.")


def print_feature_list(X):

    print_header("FINAL FEATURE LIST")

    print(f"Number of Features: {X.shape[1]}")
    print()

    for feature in X.columns:

        print(feature)


def quick_feature_report(X):

    print_header("FEATURE REPORT")

    print(f"Rows     : {X.shape[0]}")
    print(f"Columns  : {X.shape[1]}")

    numeric = len(
        X.select_dtypes(
            include=["number"]
        ).columns
    )

    categorical = len(
        X.select_dtypes(
            exclude=["number"]
        ).columns
    )

    print(f"Numeric      : {numeric}")
    print(f"Categorical  : {categorical}")

    print()

    print("First Five Features:")

    for feature in X.columns[:5]:

        print(feature)

    print()

    print("Last Five Features:")

    for feature in X.columns[-5:]:

        print(feature)


def generate_feature_selection_report(
        X,
        correlation_threshold=0.95
):
    """
    Generates every report but DOES NOT remove
    correlated variables.

    Recommended for research papers.
    """

    X_selected, summary = feature_selection_pipeline(

        X,

        correlation_threshold=correlation_threshold,

        remove_correlated=False,

        save_reports=True

    )

    save_feature_list(X_selected)

    print_feature_list(X_selected)

    quick_feature_report(X_selected)

    return X_selected, summary