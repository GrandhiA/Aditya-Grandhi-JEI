from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd
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

def generate_all_plots(model, X_test, y_test, y_prob,
                       feature_names,
                       output_folder="."):
    plot_confusion_matrix(model, X_test, y_test,
        f"{output_folder}/confusion_matrix.png")
    plot_roc_curve(model, X_test, y_test,
        f"{output_folder}/roc_curve.png")
    plot_pr_curve(model, X_test, y_test,
        f"{output_folder}/pr_curve.png")
    plot_calibration_curve(model, X_test, y_test,
        f"{output_folder}/calibration_curve.png")
    plot_probability_histogram(
        y_prob,
        f"{output_folder}/probability_histogram.png")
    plot_reliability_diagram(
        y_test,
        y_prob,
        f"{output_folder}/reliability_diagram.png")
    plot_feature_importance(
        model,
        feature_names,
        filename=f"{output_folder}/feature_importance.png")
    plot_coefficients(
        model,
        feature_names,
        filename=f"{output_folder}/coefficients.png")
    plot_permutation_importance(
        model,
        X_test,
        y_test,
        feature_names,
        filename=f"{output_folder}/permutation_importance.png")
    print("Saved all figures.")

try:
    import shap
except ImportError:
    shap = None

import matplotlib.pyplot as plt


def shap_values(model, X_train, X_test):
    """
    Compute SHAP values for a trained model.
    """

    if shap is None:
        print("SHAP is not installed.")
        return None, None

    estimator = model.model

    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
        values = explainer(X_test)

        if len(values.shape) == 3:
            values = values[:, :, 1]

    else:
        background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
        explainer = shap.Explainer(estimator.predict, background)
        values = explainer(X_test)

    return explainer, values


def plot_shap_summary(model, X_train, X_test,
                      feature_names,
                      filename="shap_summary.png"):

    if shap is None:
        return

    _, values = shap_values(model, X_train, X_test)

    plt.figure(figsize=(10,8))

    shap.plots.beeswarm(
        values,
        max_display=20,
        show=False
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_shap_bar(model, X_train, X_test,
                  feature_names,
                  filename="shap_bar.png"):

    if shap is None:
        return

    _, values = shap_values(model, X_train, X_test)

    plt.figure(figsize=(10,8))

    shap.plots.bar(
        values,
        max_display=20,
        show=False
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def explain_patient(model, X_train, X_test,
                    patient_index=0,
                    filename="patient_explanation.png"):
    """
    SHAP waterfall plot for one patient.
    """

    if shap is None:
        return

    _, values = shap_values(model, X_train, X_test)

    try:
        shap.plots.waterfall(
            values[patient_index],
            show=False
        )

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    except Exception:
        pass


def generate_shap_report(model,
                         X_train,
                         X_test,
                         feature_names,
                         output_folder="."):
    """
    Generate every SHAP visualization.
    """

    if shap is None:
        print("Skipping SHAP (package not installed).")
        return

    plot_shap_summary(
        model,
        X_train,
        X_test,
        feature_names,
        f"{output_folder}/shap_summary.png"
    )

    plot_shap_bar(
        model,
        X_train,
        X_test,
        feature_names,
        f"{output_folder}/shap_bar.png"
    )

    explain_patient(
        model,
        X_train,
        X_test,
        patient_index=0,
        filename=f"{output_folder}/patient_explanation.png"
    )
    export_shap_values(
        model,
        X_train,
        X_test,
        feature_names,
        f"{output_folder}/shap_values.csv"
    )

    export_top_shap_features(
        model,
        X_train,
        X_test,
        feature_names,
        f"{output_folder}/patient_top_features.csv"
    )
    print("Saved SHAP explanations.")

def export_shap_values(model,
                       X_train,
                       X_test,
                       feature_names,
                       filename="shap_values.csv"):
    """
    Export SHAP values for every patient.
    """

    if shap is None:
        return

    _, values = shap_values(model, X_train, X_test)

    if hasattr(values, "values"):
        shap_matrix = values.values
    else:
        shap_matrix = values

    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, 1]

    df = pd.DataFrame(
        shap_matrix,
        columns=feature_names
    )

    df.insert(0, "Patient", np.arange(len(df)))

    df.to_csv(
        filename,
        index=False
    )

    print(f"Saved SHAP values to '{filename}'.")


def export_top_shap_features(model,
                             X_train,
                             X_test,
                             feature_names,
                             filename="patient_top_features.csv",
                             top_n=5):
    """
    Export the top contributing features for every patient.
    """

    if shap is None:
        return

    _, values = shap_values(model, X_train, X_test)

    if hasattr(values, "values"):
        shap_matrix = values.values
    else:
        shap_matrix = values

    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, 1]

    rows = []

    for patient in range(shap_matrix.shape[0]):

        patient_values = shap_matrix[patient]

        order = np.argsort(np.abs(patient_values))[::-1][:top_n]

        row = {"Patient": patient}

        for i, idx in enumerate(order):

            row[f"Feature_{i+1}"] = feature_names[idx]
            row[f"SHAP_{i+1}"] = patient_values[idx]

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        filename,
        index=False
    )

    print(f"Saved patient explanations to '{filename}'.")


def patient_explanation(model,
                        X_train,
                        X_test,
                        feature_names,
                        patient_index):
    """
    Return the explanation for one patient.
    """

    if shap is None:
        return None

    _, values = shap_values(model, X_train, X_test)

    if hasattr(values, "values"):
        shap_matrix = values.values
    else:
        shap_matrix = values

    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, 1]

    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP": shap_matrix[patient_index]
    })

    df["Absolute"] = df["SHAP"].abs()

    return (
        df.sort_values(
            "Absolute",
            ascending=False
        )
        .drop(columns="Absolute")
        .reset_index(drop=True)
    )