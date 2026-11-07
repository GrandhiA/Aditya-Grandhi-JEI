import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)

import matplotlib.pyplot as plt

from utils import print_header
from plotting import generate_all_plots

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate all standard binary classification metrics.
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": recall_score(y_true, y_pred, pos_label=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR AUC": average_precision_score(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob),
        "Log Loss": log_loss(y_true, y_prob),
        "Matthews CC": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred)
    }

    return metrics


def print_metrics(metrics):
    """
    Pretty-print metrics.
    """

    print_header("MODEL PERFORMANCE")

    for metric, value in metrics.items():
        print(f"{metric:<20}: {value:.4f}")


def get_classification_report(y_true, y_pred):
    """
    Return classification report as text.
    """

    return classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )


def print_classification_report(y_true, y_pred):
    """
    Print classification report.
    """

    print_header("CLASSIFICATION REPORT")

    print(
        get_classification_report(
            y_true,
            y_pred
        )
    )


def get_confusion_matrix(y_true, y_pred):
    """
    Return confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )


def confusion_matrix_stats(cm):
    """
    Compute statistics from confusion matrix.
    """

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr
    }


def print_confusion_matrix_stats(stats):
    """
    Print confusion matrix statistics.
    """

    print_header("CONFUSION MATRIX STATISTICS")

    for key, value in stats.items():

        if isinstance(value, int):

            print(f"{key:<25}: {value}")

        else:

            print(f"{key:<25}: {value:.4f}")


def metrics_dataframe(metrics):
    """
    Convert metrics dictionary to DataFrame.
    """

    return pd.DataFrame({
        "Metric": metrics.keys(),
        "Value": metrics.values()
    })


def save_metrics(metrics, filename="metrics.csv"):
    """
    Save metrics to CSV.
    """

    metrics_dataframe(metrics).to_csv(
        filename,
        index=False
    )

    print(f"Saved metrics to '{filename}'.")


def save_predictions(
        y_true,
        y_pred,
        y_prob,
        filename="predictions.csv"
):
    """
    Save patient predictions.
    """

    predictions = pd.DataFrame({

        "Actual": y_true,

        "Predicted": y_pred,

        "Probability": y_prob

    })

    predictions.to_csv(
        filename,
        index=False
    )

    print(f"Saved predictions to '{filename}'.")
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import roc_curve

def cross_validate(model, X, y, cv=5, scoring="roc_auc"):
    """Perform stratified cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model.model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    print_header("CROSS VALIDATION")
    print(f"Metric : {scoring}")
    print(f"Scores : {np.round(scores,4)}")
    print(f"Mean   : {scores.mean():.4f}")
    print(f"Std    : {scores.std():.4f}")
    return scores

def bootstrap_ci(y_true, y_pred, y_prob, n_boot=1000, alpha=0.95):
    """Bootstrap confidence intervals for major metrics."""
    rng = np.random.default_rng(42)
    metrics = {
        "Accuracy":[],
        "Precision":[],
        "Recall":[],
        "F1":[],
        "ROC_AUC":[],
        "PR AUC":[],
        "Brier":[]
    }

    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = np.asarray(y_true)[idx]
        yp = np.asarray(y_pred)[idx]
        pr = np.asarray(y_prob)[idx]

        if len(np.unique(yt)) < 2:
            continue

        metrics["Accuracy"].append(accuracy_score(yt, yp))
        metrics["Precision"].append(precision_score(yt, yp, zero_division=0))
        metrics["Recall"].append(recall_score(yt, yp, zero_division=0))
        metrics["F1"].append(f1_score(yt, yp, zero_division=0))
        metrics["ROC_AUC"].append(roc_auc_score(yt, pr))
        metrics["PR AUC"].append(average_precision_score(yt, pr))
        metrics["Brier"].append(brier_score_loss(yt, pr))

    ci = {}
    low = (1-alpha)/2*100
    high = (1+alpha)/2*100

    for metric, values in metrics.items():
        ci[metric] = (
            np.percentile(values, low),
            np.percentile(values, high)
        )

    return ci

def print_bootstrap_ci(ci):
    print_header("95% BOOTSTRAP CONFIDENCE INTERVALS")
    for metric, (lo, hi) in ci.items():
        print(f"{metric:<12}: [{lo:.4f}, {hi:.4f}]")



def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE)
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0, 1, n_bins + 1)

    binids = np.digitize(y_prob, bins) - 1

    ece = 0

    for i in range(n_bins):

        mask = binids == i

        if np.sum(mask) == 0:
            continue

        accuracy = np.mean(y_true[mask])

        confidence = np.mean(y_prob[mask])

        weight = np.mean(mask)

        ece += weight * abs(accuracy - confidence)

    return ece



def optimal_threshold(y_true, y_prob):
    """Maximum Youden Index."""
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    j = tpr - fpr
    i = np.argmax(j)
    return thresh[i], tpr[i], 1-fpr[i]

def evaluate_threshold(y_true, y_prob):
    """Evaluate best classification threshold."""
    threshold, sens, spec = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    print_header("OPTIMAL THRESHOLD")
    print(f"Threshold   : {threshold:.4f}")
    print(f"Sensitivity : {sens:.4f}")
    print(f"Specificity : {spec:.4f}")

    return threshold, metrics


def evaluate_calibration(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    print_header("CALIBRATION COMPARISON")

    models = {

        "Uncalibrated": model,

        "Platt Scaling":
            CalibratedClassifierCV(
                model,
                method="sigmoid",
                cv=5
            ),

        "Isotonic":
            CalibratedClassifierCV(
                model,
                method="isotonic",
                cv=5
            )

    }

    plt.figure(figsize=(6,6))

    plt.plot(
        [0,1],
        [0,1],
        "--",
        color="black",
        label="Perfect Calibration"
    )

    results = {}

    for name, clf in models.items():

        clf.fit(
            X_train,
            y_train
        )

        probs = clf.predict_proba(X_test)[:,1]
        
        prob_true, prob_pred = calibration_curve(
            y_test,
            probs,
            n_bins=10,
            strategy="quantile"
        )

        plt.figure(figsize=(6,6))

        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label=name
        )

        plt.plot(
            [0,1],
            [0,1],
            "--",
            color="gray",
            label="Perfect calibration"
        )

        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title(f"Calibration Curve ({name})")
        plt.legend()

        plt.tight_layout()

        plt.savefig(f"calibration_curve_{name}.png", dpi=300)

        plt.close()

        preds = (probs >= 0.5).astype(int)

        roc = roc_auc_score(
            y_test,
            probs
        )

        pr = average_precision_score(
            y_test,
            probs
        )

        brier = brier_score_loss(
            y_test,
            probs
        )

        ll = log_loss(
            y_test,
            probs
        )

        mcc = matthews_corrcoef(
            y_test,
            preds
        )

        kappa = cohen_kappa_score(
            y_test,
            preds
        )

        ece = expected_calibration_error(
            y_test,
            probs
        )

        frac_pos, mean_pred = calibration_curve(
            y_test,
            probs,
            n_bins=10
        )

        plt.plot(
            mean_pred,
            frac_pos,
            marker="o",
            label=name
        )

        results[name] = {

            "ROC AUC": roc,

            "PR AUC": pr,

            "Brier": brier,

            "Log Loss": ll,

            "ECE": ece,

            "Matthews CC": mcc,

            "Cohen Kappa": kappa

        }

    plt.xlabel("Mean Predicted Probability")

    plt.ylabel("Observed Frequency")

    plt.title("Calibration Curve")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "calibration_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()

    for name, metrics in results.items():

        print(name)

        print("-"*40)

        for key, value in metrics.items():

            print(f"{key:<15}: {value:.4f}")

        print()

    return results



def evaluate_model(model, X_train, X_test, y_train, y_test,
                   save=True,
                   metrics_file="metrics.csv",
                   predictions_file="predictions.csv"):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    metrics = calculate_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics)
    print_classification_report(y_test, y_pred)

    cm = get_confusion_matrix(y_test, y_pred)
    print_confusion_matrix_stats(confusion_matrix_stats(cm))

    cv_scores = cross_validate(model, X_train, y_train)

    ci = bootstrap_ci(y_test, y_pred, y_prob)
    print_bootstrap_ci(ci)

    threshold, threshold_metrics = evaluate_threshold(y_test, y_prob)


    calibration_results = evaluate_calibration(
        model.model,
        X_train,
        y_train,
        X_test,
        y_test
    )


    if save:
        save_metrics(metrics, metrics_file)
        save_predictions(y_test, y_pred, y_prob, predictions_file)

    return {
        "metrics": metrics,
        "cv_scores": cv_scores,
        "bootstrap_ci": ci,
        "threshold": threshold,
        "calibration": calibration_results,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_prob
    }