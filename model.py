import numpy as np
from sklearn.linear_model import LogisticRegression
from config import RANDOM_STATE
from utils import print_header
import joblib
from config import config
from sklearn.calibration import CalibratedClassifierCV


class BaseModel:
    """
    Base class for every model.
    """

    def __init__(self):

        self.model = None

    def fit(self, X_train, y_train):

        self.model.fit(X_train, y_train)

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    def coefficients(self):

        if hasattr(self.model, "coef_"):

            return self.model.coef_[0]

        return None

    def feature_importance(self):

        if hasattr(self.model, "feature_importances_"):

            return self.model.feature_importances_

        return None


class LogisticModel(BaseModel):

    """
    Standard Logistic Regression.
    """

    def __init__(

        self,

        penalty="l2",

        C=1.0,

        class_weight="balanced",

        max_iter=5000,

        solver="lbfgs"

    ):

        super().__init__()

        self.model = LogisticRegression(

            penalty=penalty,

            C=C,

            class_weight=class_weight,

            max_iter=max_iter,

            solver=solver,

            random_state=RANDOM_STATE

        )

        print_header("LOGISTIC REGRESSION")

        print(f"Penalty      : {penalty}")
        print(f"C            : {C}")
        print(f"Solver       : {solver}")
        print(f"Max Iter     : {max_iter}")
        print(f"Class Weight : {class_weight}")

        print()

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


class RandomForestModel(BaseModel):

    def __init__(
            self,
            n_estimators=600,
            max_depth=20,
            min_samples_split=18,
            min_samples_leaf=6,
            max_features=4,
            class_weight="balanced"
    ):

        super().__init__()

        self.model = RandomForestClassifier(

            n_estimators=n_estimators,

            max_depth=max_depth,

            min_samples_split=min_samples_split,

            min_samples_leaf=min_samples_leaf,


            criterion="gini",
            max_leaf_nodes=256,
            ccp_alpha=6.346854071267571e-05,



            max_features=max_features,

            class_weight=class_weight,

            random_state=RANDOM_STATE,

            n_jobs=-1

        )

        print_header("RANDOM FOREST")

        print(f"Trees : {n_estimators}")
        print(f"Depth : {max_depth}")
        print(f"Max Features : {max_features}")
        print()

class XGBoostModel(BaseModel):

    def __init__(

            self,

            n_estimators=500,

            learning_rate=0.05,

            max_depth=4

    ):

        super().__init__()

        if XGBClassifier is None:

            raise ImportError(

                "Install xgboost first."

            )

        self.model = XGBClassifier(

            n_estimators=n_estimators,

            learning_rate=learning_rate,

            max_depth=max_depth,

            random_state=RANDOM_STATE,

            eval_metric="logloss"

        )

        print_header("XGBOOST")


class LightGBMModel(BaseModel):

    def __init__(

            self,

            n_estimators=500,

            learning_rate=0.05,

            max_depth=-1

    ):

        super().__init__()

        if LGBMClassifier is None:

            raise ImportError(

                "Install lightgbm first."

            )

        self.model = LGBMClassifier(

            n_estimators=n_estimators,

            learning_rate=learning_rate,

            max_depth=max_depth,

            random_state=RANDOM_STATE

        )

        print_header("LIGHTGBM")


class CatBoostModel(BaseModel):

    def __init__(

            self,

            iterations=500,

            learning_rate=0.05,

            depth=6

    ):

        super().__init__()

        if CatBoostClassifier is None:

            raise ImportError(

                "Install catboost first."

            )

        self.model = CatBoostClassifier(

            iterations=iterations,

            learning_rate=learning_rate,

            depth=depth,

            random_seed=RANDOM_STATE,

            verbose=False

        )

        print_header("CATBOOST")



def create_model(model_name, **kwargs):
    """
    Create a model by name.

    Supported models
    ----------------
    logistic
    random_forest (or rf)
    xgboost (or xgb)
    lightgbm (or lgbm)
    catboost
    """

    model_name = model_name.lower()

    if model_name == "logistic":

        return LogisticModel(**kwargs)

    elif model_name in ["random_forest", "rf"]:

        return RandomForestModel(**kwargs)

    elif model_name in ["xgboost", "xgb"]:

        return XGBoostModel(**kwargs)

    elif model_name in ["lightgbm", "lgbm"]:

        return LightGBMModel(**kwargs)

    elif model_name == "catboost":

        return CatBoostModel(**kwargs)

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


def train_model(
    model,
    X_train,
    y_train
):
    """
    Train any model.
    """

    print_header("TRAINING MODEL")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    return model


def predict_model(
    model,
    X
):
    """
    Predict class labels.
    """

    return model.predict(X)


def predict_probabilities(
    model,
    X
):
    """
    Predict positive-class probabilities.
    """

    return model.predict_proba(X)[:, 1]


def predict_all(
    model,
    X
):
    """
    Return predictions and probabilities.
    """

    y_pred = predict_model(
        model,
        X
    )

    y_prob = predict_probabilities(
        model,
        X
    )

    return y_pred, y_prob


def save_model(
    model,
    filename="trained_model.pkl"
):
    """
    Save a trained model.
    """

    joblib.dump(
        model,
        filename
    )

    print(f"Saved model to '{filename}'.")


def load_model(
    filename="trained_model.pkl"
):
    """
    Load a saved model.
    """

    model = joblib.load(filename)

    print(f"Loaded model from '{filename}'.")

    return model


def summarize_model(model):
    """
    Print model information.
    """

    print_header("MODEL SUMMARY")

    print(type(model.model))

    print()

    print(model.model)


def available_models():
    """
    Print supported models.
    """

    print_header("AVAILABLE MODELS")

    print("logistic")
    print("random_forest (rf)")
    print("xgboost (xgb)")
    print("lightgbm (lgbm)")
    print("catboost")