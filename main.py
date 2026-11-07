# ============================================================
# MAIN.PY
# ============================================================

from config import *

from data_loader import (
    load_dataset,
    split_features_target
)

from preprocessing import (
    preprocess_features,
    preprocessing_pipeline
)

from model import (
    create_model
)

from evaluation import (
    evaluate_model
)

from plotting import (
    generate_all_plots
)

from explainability import (
    generate_shap_report
)

from reporting import (
    report_results,
    comparison_table,
    comparison_excel,
    leaderboard,
    print_best_model
)

from threshold_analysis import threshold_analysis

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

DATASET = "New_set.csv"

TARGETS = [
    "Mortality"
]

MODELS = [
"rf"
]

all_results = []

# ============================================================
# RUN EXPERIMENTS
# ============================================================

for target in TARGETS:

    print("\n" + "=" * 70)
    print(f"TARGET: {target}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset(DATASET)

    X = df.drop(columns=[target])

    y = df[target]

    # --------------------------------------------------------
    # Feature preprocessing
    # --------------------------------------------------------

    X, preprocessing_metadata = preprocess_features(
        X,
        config
    )

    feature_names = list(X.columns)

    # --------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------
    df = load_dataset(DATASET)
    X, y = split_features_target(
        df,
        TARGET,
        drop_columns=DROP_COLUMNS,
        patient_id=PATIENT_ID_COLUMN
    )
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    # --------------------------------------------------------
    # Scaling / alignment
    # --------------------------------------------------------

    X_train, X_test, scaler = preprocessing_pipeline(
        X_train,
        X_test,
        config
    )

    feature_names = list(X_train.columns)

    # --------------------------------------------------------
    # Train every model
    # --------------------------------------------------------

    for model_name in MODELS:

        print("\n" + "-" * 60)
        print(f"MODEL: {model_name.upper()}")
        print("-" * 60)

        model = create_model(model_name)

        results = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )       
        # ============================================================
        # FINAL ISOTONIC CALIBRATION
        # ============================================================

        from sklearn.calibration import CalibratedClassifierCV

        print("\nTraining final isotonic calibration...\n")

        isotonic_model = CalibratedClassifierCV(
            model.model,
            method="isotonic",
            cv=5
        )

        isotonic_model.fit(
            X_train,
            y_train
        )

        print("Done.\n")
        # ============================================================
        # FINAL CALIBRATED MODEL PERFORMANCE
        # ============================================================

        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            brier_score_loss,
            log_loss,
            matthews_corrcoef,
            cohen_kappa_score
        )

        print("="*60)
        print("FINAL ISOTONIC MODEL")
        print("="*60)

        y_prob = isotonic_model.predict_proba(X_test)[:,1]

        y_pred = (y_prob >= 0.5).astype(int)

        print(f"ROC AUC      : {roc_auc_score(y_test,y_prob):.4f}")
        print(f"PR AUC       : {average_precision_score(y_test,y_prob):.4f}")
        print(f"Brier Score  : {brier_score_loss(y_test,y_prob):.4f}")
        print(f"Log Loss     : {log_loss(y_test,y_prob):.4f}")
        print(f"Matthews CC  : {matthews_corrcoef(y_test,y_pred):.4f}")
        print(f"Cohen Kappa  : {cohen_kappa_score(y_test,y_pred):.4f}")
        # ============================================================
        # SAVE FINAL MODEL FOR STREAMLIT
        # ============================================================

        
        import joblib

        model_bundle = {

            # sklearn RandomForestClassifier
            "model": isotonic_model,

            # StandardScaler
            "scaler": scaler,

            # Feature order used during training
            "features": feature_names,

            # Model metadata
            "model_name": "Random Forest",

            "version": "1.0",

            "roc_auc": results["metrics"]["ROC_AUC"],
  
            "accuracy": results["metrics"]["Accuracy"],

            "brier_score": results["metrics"]["Brier Score"],

            "training_samples": len(X_train),

            "test_samples": len(X_test)

        }

        joblib.dump(

            model_bundle,
        
            "sepsis_dashboard.pkl"

        )
        

        # ============================================================
        # VERIFY SAVED MODEL
        # ============================================================
        
        loaded = joblib.load("sepsis_dashboard.pkl")

        print("\nIn-memory probability:")
        p1 = model.model.predict_proba(X_test.iloc[[0]])[0,1]
        print(p1)

        print("\nLoaded probability:")
        p2 = loaded["model"].predict_proba(X_test.iloc[[0]])[0,1]
        print(p2)

        print("\nDifference:", abs(p1 - p2))

        print("\nSaved dashboard model to 'sepsis_dashboard.pkl'")
        
        # ----------------------------------------------------
        # Save reports
        # ----------------------------------------------------

        folder = report_results(

            model_name=model_name,

            target=target,

            metrics=results["metrics"],

            cv_scores=results["cv_scores"],

            bootstrap_ci=results["bootstrap_ci"],

            y_true=y_test,

            y_pred=results["predictions"],

            y_prob=results["probabilities"]
        )
   
        # ----------------------------------------------------
        # Threshold Optimization
        # ----------------------------------------------------

        threshold_results = threshold_analysis(

            y_true=y_test,

            y_prob=results["probabilities"],

            output_file=f"{folder}/threshold_analysis.csv"

        )

        # ----------------------------------------------------
        # Figures
        # ----------------------------------------------------

        generate_all_plots(

            model=model,

            X_test=X_test,

            y_test=y_test,

            y_prob=results["probabilities"],

            feature_names=feature_names,

            output_folder=folder
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        generate_shap_report(

            model=model,

            X_train=X_train,

            X_test=X_test,

            feature_names=feature_names,

            output_folder=folder
        )

        # ----------------------------------------------------
        # Store for comparison
        # ----------------------------------------------------

        all_results.append({

            "model": model_name,

            "target": target,

            "metrics": results["metrics"]

        })

# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("OVERALL MODEL COMPARISON")
print("=" * 70)

comparison_table(
    all_results,
    filename="comparison.csv"
)

print(all_results)
print(len(all_results))

comparison_excel(
    all_results,
    filename="comparison.xlsx"
)

leaderboard(all_results)

print_best_model(all_results)

print("\nFinished all experiments.")