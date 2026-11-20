import optuna
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import json
from pathlib import Path
from src.lab.data import load_data, build_preprocessor
from src.lab.model import build_model


def objective(trial):

    X, y = load_data()

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }

    model = build_model("gbm")
    model.set_params(**params)

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    return scores.mean()


def main():

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=40)

    print("Best params:", study.best_params)
    print("Best score:", study.best_value)
    # save params
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    with open("outputs/reports/best_params_gbm.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

if __name__ == "__main__":
    main()