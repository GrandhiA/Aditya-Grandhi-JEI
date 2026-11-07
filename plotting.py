import matplotlib
matplotlib.use("Agg")   # Non-GUI backend
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibrationDisplay

from utils import print_header

def plot_confusion_matrix(model, X_test, y_test,
                          filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay.from_estimator(
        model.model,
        X_test,
        y_test,
        cmap="Blues",
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def plot_roc_curve(model, X_test, y_test,
                   filename="roc_curve.png"):
    fig, ax = plt.subplots(figsize=(6,6))
    RocCurveDisplay.from_estimator(
        model.model,
        X_test,
        y_test,
        ax=ax
    )
    ax.plot([0,1],[0,1],"k--")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(filename,dpi=300)
    plt.close(fig)

def plot_pr_curve(model, X_test, y_test,
                  filename="pr_curve.png"):
    fig, ax = plt.subplots(figsize=(6,6))
    PrecisionRecallDisplay.from_estimator(
        model.model,
        X_test,
        y_test,
        ax=ax
    )
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(filename,dpi=300)
    plt.close(fig)

def plot_calibration_curve(model, X_test, y_test,
                           filename="calibration_curve.png"):
    fig, ax = plt.subplots(figsize=(6,6))
    CalibrationDisplay.from_estimator(
        model.model,
        X_test,
        y_test,
        n_bins=10,
        strategy="quantile",
        ax=ax
    )
    ax.set_title("Calibration Curve")
    fig.tight_layout()
    fig.savefig(filename,dpi=300)
    plt.close(fig)

from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

def plot_probability_histogram(y_prob, filename="probability_histogram.png"):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.hist(y_prob, bins=20, edgecolor="black")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Patients")
    ax.set_title("Predicted Risk Distribution")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def plot_feature_importance(model, feature_names,
                            top_n=20,
                            filename="feature_importance.png"):
    if not hasattr(model.model, "feature_importances_"):
        return

    imp = pd.Series(
        model.model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8,6))
    imp.iloc[::-1].plot.barh(ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def plot_coefficients(model, feature_names,
                      top_n=20,
                      filename="coefficients.png"):
    if not hasattr(model.model, "coef_"):
        return

    coef = pd.Series(
        model.model.coef_[0],
        index=feature_names
    )

    coef = coef.reindex(
        coef.abs().sort_values(ascending=False).index
    ).head(top_n)

    fig, ax = plt.subplots(figsize=(8,6))
    coef.iloc[::-1].plot.barh(ax=ax)
    ax.set_title(f"Top {top_n} Logistic Coefficients")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def plot_permutation_importance(model, X_test, y_test,
                                feature_names,
                                top_n=20,
                                filename="permutation_importance.png"):
    perm = permutation_importance(
        model.model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )

    imp = pd.Series(
        perm.importances_mean,
        index=feature_names
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8,6))
    imp.iloc[::-1].plot.barh(ax=ax)
    ax.set_title("Permutation Importance")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def plot_reliability_diagram(y_true, y_prob,
                             filename="reliability_diagram.png"):
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot([0,1],[0,1],"k--")
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Reliability Diagram")
    fig.tight_layout()
    fig.savefig(filename,dpi=300)
    plt.close(fig)


def generate_all_plots(
    model,
    X_test,
    y_test,
    y_prob,
    feature_names,
    output_folder="."
):
    """
    Generate every evaluation figure.
    """

    plot_confusion_matrix(
        model,
        X_test,
        y_test,
        f"{output_folder}/confusion_matrix.png"
    )

    plot_roc_curve(
        model,
        X_test,
        y_test,
        f"{output_folder}/roc_curve.png"
    )

    plot_pr_curve(
        model,
        X_test,
        y_test,
        f"{output_folder}/pr_curve.png"
    )

    plot_calibration_curve(
        model,
        X_test,
        y_test,
        f"{output_folder}/calibration_curve.png"
    )

    plot_probability_histogram(
        y_prob,
        f"{output_folder}/probability_histogram.png"
    )

    plot_reliability_diagram(
        y_test,
        y_prob,
        f"{output_folder}/reliability_diagram.png"
    )

    plot_feature_importance(
        model,
        feature_names,
        filename=f"{output_folder}/feature_importance.png"
    )

    plot_coefficients(
        model,
        feature_names,
        filename=f"{output_folder}/coefficients.png"
    )

    plot_permutation_importance(
        model,
        X_test,
        y_test,
        feature_names,
        filename=f"{output_folder}/permutation_importance.png"
    )

    print_header("PLOTS")
    print("Saved all evaluation figures.")
