import os
from datetime import datetime
import pandas as pd


def create_experiment_folder(model_name,
                             target,
                             root="Results"):
    """
    Create a timestamped experiment folder.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder = os.path.join(
        root,
        f"{target}_{model_name}_{timestamp}"
    )

    os.makedirs(folder, exist_ok=True)

    return folder


def save_metrics(metrics, folder):
    """
    Save metrics dictionary.
    """

    pd.DataFrame({
        "Metric": metrics.keys(),
        "Value": metrics.values()
    }).to_csv(
        os.path.join(folder, "metrics.csv"),
        index=False
    )


def save_predictions(y_true,
                     y_pred,
                     y_prob,
                     folder):
    """
    Save predictions.
    """

    pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred,
        "Probability": y_prob
    }).to_csv(
        os.path.join(folder, "predictions.csv"),
        index=False
    )


def save_feature_table(df,
                       filename,
                       folder):
    """
    Save feature table.
    """

    if df is None:
        return

    df.to_csv(
        os.path.join(folder, filename),
        index=False
    )


def write_summary(model_name,
                  target,
                  metrics,
                  folder):
    """
    Write experiment summary.
    """

    path = os.path.join(
        folder,
        "experiment_summary.txt"
    )

    with open(path, "w") as f:

        f.write("="*60 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("="*60 + "\n\n")

        f.write(f"Model : {model_name}\n")
        f.write(f"Target: {target}\n")
        f.write(f"Date  : {datetime.now()}\n\n")

        f.write("Performance\n")
        f.write("-"*40 + "\n")

        for metric, value in metrics.items():
            f.write(f"{metric:<20}: {value:.4f}\n")


def save_cv_scores(scores, folder):
    """
    Save cross-validation scores.
    """

    pd.DataFrame({
        "Fold": range(1, len(scores)+1),
        "ROC_AUC": scores
    }).to_csv(
        os.path.join(folder, "cross_validation.csv"),
        index=False
    )


def save_bootstrap(ci, folder):
    """
    Save bootstrap confidence intervals.
    """

    rows = []

    for metric, (low, high) in ci.items():

        rows.append({
            "Metric": metric,
            "Lower95CI": low,
            "Upper95CI": high
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(folder, "bootstrap_ci.csv"),
        index=False
    )


def report_results(model_name,
                   target,
                   metrics,
                   cv_scores,
                   bootstrap_ci,
                   y_true,
                   y_pred,
                   y_prob,
                   coefficients=None,
                   feature_importance=None,
                   permutation=None,
                   folder=None):
    """
    Save every result from an experiment.
    """

    if folder is None:

        folder = create_experiment_folder(
            model_name,
            target
        )

    save_metrics(metrics, folder)

    save_predictions(
        y_true,
        y_pred,
        y_prob,
        folder
    )

    save_cv_scores(
        cv_scores,
        folder
    )

    save_bootstrap(
        bootstrap_ci,
        folder
    )

    save_feature_table(
        coefficients,
        "coefficients.csv",
        folder
    )

    save_feature_table(
        feature_importance,
        "feature_importance.csv",
        folder
    )

    save_feature_table(
        permutation,
        "permutation_importance.csv",
        folder
    )

    write_summary(
        model_name,
        target,
        metrics,
        folder
    )

    print(f"\nResults saved to:\n{folder}")

    return folder

def comparison_table(results, filename="comparison.csv"):
    """
    Create a comparison table for multiple models.
    """

    rows = []

    for r in results:

        metrics = r["metrics"]

        rows.append({
            "Model": r["model"],
            "Target": r["target"],
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1 Score"],
            "ROC_AUC": metrics["ROC_AUC"],
            "PR AUC": metrics["PR AUC"],
            "Brier": metrics["Brier Score"]
        })

    df = pd.DataFrame(rows)

    df.sort_values(
        "ROC_AUC",
        ascending=False,
        inplace=True
    )

    df.to_csv(filename,index=False)

    return df


def comparison_excel(results,
                     filename="comparison.xlsx"):
    """
    Save comparison table as Excel.
    """

    df = comparison_table(results)

    df.to_excel(
        filename,
        index=False
    )

    return df

def print_best_model(results):
    """
    Print the best-performing model.
    """

    best = max(
        results,
        key=lambda x: x["metrics"]["ROC_AUC"]
    )

    print("\n==============================")
    print("BEST MODEL")
    print("==============================")
    print(f"Model : {best['model']}")
    print(f"Target: {best['target']}")
    print(f"ROC_AUC: {best['metrics']['ROC_AUC']:.4f}")

    return best

def leaderboard(results,
                metric="ROC_AUC"):
    """
    Print model leaderboard.
    """

    df = comparison_table(results)

    df = df.sort_values(
        metric,
        ascending=False
    )

    print("\nMODEL LEADERBOARD\n")
    print(df)

    return df