import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70

MODEL_TYPES = {
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
}


def _mlflow_params(params: dict) -> dict:
    return {
        key: json.dumps(value) if isinstance(value, (dict, list)) else value
        for key, value in params.items()
    }


def _load_train_data(data_path: str, extra_data_paths: list[str]) -> pd.DataFrame:
    dataframes = [pd.read_csv(data_path)]

    for path in extra_data_paths:
        dataframes.append(pd.read_csv(path))

    return pd.concat(dataframes, ignore_index=True)


def _build_model(params: dict):
    model_type = params.get("model_type", "random_forest")

    if model_type not in MODEL_TYPES:
        supported = ", ".join(sorted(MODEL_TYPES))
        raise ValueError(f"Unsupported model_type '{model_type}'. Supported: {supported}")

    model_params = {
        key: value
        for key, value in params.items()
        if key not in {"model_type", "extra_data_paths"}
    }
    model_params.setdefault("random_state", 42)

    return MODEL_TYPES[model_type](**model_params)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "Default"))

    # TODO 1: Doc du lieu huan luyen va danh gia
    extra_data_paths = params.get("extra_data_paths", [])
    df_train = _load_train_data(data_path, extra_data_paths)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(_mlflow_params(params))
        mlflow.log_param("train_rows", len(df_train))

        # TODO 4: Khoi tao va huan luyen mo hinh
        model = _build_model(params)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)

        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)

        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "passed_threshold": acc >= EVAL_THRESHOLD,
                },
                f,
                indent=4,
            )

        # TODO 9: Luu mo hinh ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train(params)
