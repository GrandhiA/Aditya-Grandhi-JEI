import pandas as pd
import optuna

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Settings

DATASET = "New_set.csv"
TARGET = "Mortality"

RANDOM_STATE = 42

N_TRIALS = 100

# Loading data

df = pd.read_csv(DATASET)

X = df.drop(columns=[TARGET])
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

# CV

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


def objective(trial):

    model = RandomForestClassifier(

        n_estimators=trial.suggest_int(
            "n_estimators",
            100,
            600
        ),

        max_depth=trial.suggest_int(
            "max_depth",
            5,
            40
        ),

        min_samples_split=trial.suggest_int(
            "min_samples_split",
            2,
            30
        ),

        min_samples_leaf=trial.suggest_int(
            "min_samples_leaf",
            1,
            15
        ),

        max_features=trial.suggest_float(
            "max_features",
            0.2,
            1.0
        ),

        criterion=trial.suggest_categorical(
            "criterion",
            ["gini", "entropy", "log_loss"]
        ),

        bootstrap=trial.suggest_categorical(
            "bootstrap",
            [True, False]
        ),

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    return scores.mean()


# Running optuna


study = optuna.create_study(
    direction="maximize",
    study_name="RandomForest Optimization"
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)

# Printing best parameters


print("\n")
print("="*60)
print("BEST PARAMETERS")
print("="*60)

print(study.best_params)

print()

print("Best CV ROC AUC")
print(round(study.best_value,4))

# Training final model


best_model = RandomForestClassifier(

    **study.best_params,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1
)

best_model.fit(
    X_train,
    y_train
)



y_prob = best_model.predict_proba(X_test)[:,1]

y_pred = best_model.predict(X_test)

roc = roc_auc_score(
    y_test,
    y_prob
)

acc = accuracy_score(
    y_test,
    y_pred
)

prec = precision_score(
    y_test,
    y_pred
)

rec = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()

spec = tn/(tn+fp)

print("\n")
print("="*60)
print("TEST SET PERFORMANCE")
print("="*60)

print(f"ROC AUC      : {roc:.4f}")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"Specificity  : {spec:.4f}")
print(f"F1 Score     : {f1:.4f}")

# Save results


trials = study.trials_dataframe()

trials.to_csv(
    "optuna_results.csv",
    index=False
)

print("\nSaved all trials to optuna_results.csv")