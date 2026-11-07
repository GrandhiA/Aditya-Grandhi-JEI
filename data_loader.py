import os
import pandas as pd

from utils import (
    print_header,
    dataset_summary
)

def load_dataset(filename, print_summary=True):

    print_header("LOADING DATASET")

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"\nDataset not found:\n{filename}"
        )

    extension = os.path.splitext(filename)[1].lower()

    if extension == ".csv":

        df = pd.read_csv(filename)

    elif extension in [".xlsx", ".xls"]:

        df = pd.read_excel(filename)

    else:

        raise ValueError(
            "Dataset must be CSV or Excel."
        )

    print(f"Loaded dataset successfully.")
    print(f"File : {filename}")
    print(f"Shape: {df.shape}")

    if print_summary:
        dataset_summary(df)

    # Remove completely empty rows
    before = len(df)
    df = df.dropna(how="all")
    after = len(df)
    print(f"Removed {before-after} completely empty rows.")
    return df


def verify_target(df, target):

    if target not in df.columns:
        raise ValueError(
            f"\nTarget column '{target}' not found."
        )
    print(f"\nTarget column verified: {target}")


def verify_patient_id(df, patient_id_column):

    if patient_id_column is None:
        return

    if patient_id_column not in df.columns:
        raise ValueError(
            f"\nPatient ID column '{patient_id_column}' not found."
        )

    duplicates = df[patient_id_column].duplicated().sum()

    if duplicates > 0:
        print(
            f"Warning: {duplicates} duplicated patient IDs."
        )
    else:
        print("Patient IDs verified.")


def check_duplicate_rows(df):

    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    return duplicates


def check_duplicate_columns(df):

    duplicated = df.columns[df.columns.duplicated()]
    if len(duplicated) == 0:
        print("Duplicate columns: None")
    else:
        print("\nDuplicate Columns:")
        for col in duplicated:
            print(col)
    return list(duplicated)


def missing_value_report(df):
    print_header("MISSING VALUE REPORT")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("No missing values found.")
    else:
        report = pd.DataFrame({
            "Missing Values": missing,
            "Percent": (
                missing /
                len(df) *
                100
            ).round(2)
        })
        print(report)

    return missing


def constant_columns(df):
    constant = []
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            constant.append(col)

    if len(constant) > 0:
        print_header("CONSTANT COLUMNS")
        for col in constant:
            print(col)
    else:
        print("\nNo constant columns found.")

    return constant


def validate_dataset(df, target, patient_id=None):    
    print_header("VALIDATING DATASET")
    verify_target(df, target)
    verify_patient_id(df, patient_id)
    check_duplicate_rows(df)
    check_duplicate_columns(df)
    missing_value_report(df)
    constant_columns(df)
    print("\nDataset validation complete.")


def split_features_target(df, target, drop_columns=None, patient_id=None):
    columns_to_drop = [target]
    if drop_columns is not None:
        columns_to_drop.extend(drop_columns)

    if patient_id is not None:
        columns_to_drop.append(patient_id)

    X = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    y = df[target]
    return X, y


def feature_summary(X):
    print_header("FEATURE SUMMARY")
    print(f"Number of Features : {X.shape[1]}")
    print("\nFeature Names:\n")

    for feature in X.columns:
        print(feature)