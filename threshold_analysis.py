import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)


def threshold_analysis(
        y_true,
        y_prob,
        output_file="threshold_analysis.csv"
):

    rows = []

    thresholds = np.arange(0.05, 0.96, 0.05)

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        ).ravel()

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        specificity = tn / (tn + fp)

        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f2 = fbeta_score(
            y_true,
            y_pred,
            beta=2,
            zero_division=0
        )

        rows.append({

            "Threshold": threshold,

            "Accuracy": accuracy,

            "Precision (PPV)": precision,

            "Recall (Sensitivity)": recall,

            "Specificity": specificity,

            "Negative Predictive Value": npv,

            "F1": f1,

            "F2": f2,

            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn

        })

    df = pd.DataFrame(rows)

    df.to_csv(output_file, index=False) 

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9,6))

    metrics = [

        "Recall (Sensitivity)",

        "Precision (PPV)",

        "Negative Predictive Value",

        "Specificity",

        "F1",

        "F2"

    ]

    for metric in metrics:

        plt.plot(

            df["Threshold"],

            df[metric],

            linewidth=2,

            label=metric

        )

    plt.xlabel("Decision Threshold")

    plt.ylabel("Score")

    plt.title("Threshold Optimization")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        output_file.replace(".csv", ".png"),

        dpi=300

    )

    plt.close()

    print("\nSaved threshold analysis to")
    print(output_file)

    print("\nBest Recall")
    print(df.loc[df["Recall (Sensitivity)"].idxmax()])

    print("\nBest F2")
    print(df.loc[df["F2"].idxmax()])

    print("\nBest F1")
    print(df.loc[df["F1"].idxmax()])

    return df