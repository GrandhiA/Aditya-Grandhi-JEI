import os
import random
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from config import config

def set_random_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def print_header(title):

    line = "=" * 60

    print("\n")
    print(line)
    print(title.center(60))
    print(line)


def print_subheader(title):

    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def create_folder(path):
    """
    Create folder if it doesn't exist.
    """

    os.makedirs(path, exist_ok=True)


def create_output_structure(base_folder, target):

    timestamp = get_timestamp()

    run_folder = os.path.join(
        base_folder,
        f"{target}_{timestamp}"
    )

    folders = {

        "root": run_folder,

        "figures":
            os.path.join(run_folder, "Figures"),

        "tables":
            os.path.join(run_folder, "Tables"),

        "predictions":
            os.path.join(run_folder, "Predictions"),

        "explainability":
            os.path.join(run_folder, "Explainability"),

        "logs":
            os.path.join(run_folder, "Logs")

    }

    for folder in folders.values():
        create_folder(folder)

    return folders


def save_dataframe(df, filename):

    if df is None:
        return

    if filename.lower().endswith(".csv"):
        df.to_csv(filename, index=False)

    elif filename.lower().endswith(".xlsx"):
        df.to_excel(filename, index=False)

    else:
        raise ValueError(
            "Filename must end with .csv or .xlsx"
        )


def save_text(text, filename):

    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(text))


def dataset_summary(df):

    print_header("DATASET SUMMARY")

    print(f"Rows      : {df.shape[0]}")
    print(f"Columns   : {df.shape[1]}")

    print("\nData Types")

    print(df.dtypes.to_string())

    print("\nMissing Values")

    print(df.isna().sum().to_string())


def class_distribution(target):

    print_header("CLASS DISTRIBUTION")

    counts = target.value_counts()

    percents = target.value_counts(normalize=True) * 100

    summary = pd.DataFrame({

        "Count": counts,

        "Percent": percents.round(2)

    })

    print(summary)


def version_information():

    info = {

        "Python":
            platform.python_version(),

        "NumPy":
            np.__version__,

        "Pandas":
            pd.__version__,

        "Scikit-learn":
            sklearn.__version__

    }

    return info


def save_version_information(filename):

    info = version_information()

    lines = []

    for key, value in info.items():

        lines.append(f"{key}: {value}")

    save_text("\n".join(lines), filename)


def save_configuration(config_module, filename):

    lines = []

    for key in dir(config_module):

        if key.isupper():

            value = getattr(config_module, key)

            lines.append(f"{key} = {value}")

    save_text("\n".join(lines), filename)

def finished():

    print_header("PIPELINE FINISHED SUCCESSFULLY")