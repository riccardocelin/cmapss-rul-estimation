import json
from pathlib import Path
import os

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(PROJECT_ROOT) + "/configs/training_config.local.json"  # remember to create a new file copy from configs/training_config.template.json
# bash to start mlflow server: mlflow ui --backend-store-uri ./mlruns --artifacts-destination ./mlruns --port 8080

def flatten_dict(data, parent_key=""):
    flat = {}
    for key, value in data.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            flat[new_key] = value
    return dict(sorted(flat.items()))


def log_params(params):
    for key, value in params.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            mlflow.log_param(key, json.dumps(list(value)))
        else:
            mlflow.log_param(key, value)


def load_config(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tabular_data(cfg, is_training=True):

    label = ""
    if is_training is True:
        label = "training"
    else:
        label = "test"

    # load dataset
    data_folder = cfg["data_base_folder"] + "/" + cfg["data_type"] + "/" + cfg["data_version"]
    dataset_X_name = cfg[label]["dataset_tabular_X_name"]
    dataset_y_name = cfg[label]["dataset_tabular_y_name"]

    data_paths = {
        "data_folder": data_folder,
        "dataset_X_name": dataset_X_name,
        "dataset_y_name": dataset_y_name,
    }

    X_with_engine = pd.read_csv(data_folder + "/" + dataset_X_name + ".csv")
    y = pd.read_csv(data_folder + "/" + dataset_y_name + ".csv")

    X = X_with_engine.loc[:, X_with_engine.columns != "engine_id"]

    return X, X_with_engine, y, data_paths


def load_sequence_data(cfg, is_training=True):

    label = ""
    if is_training is True:
        label = "training"
    else:
        label = "test"

    # load dataset
    data_folder = cfg["data_base_folder"] + "/" + cfg["data_type"] + "/" + cfg["data_version"]
    dataset_X_y_name = cfg[label]["dataset_sequence_X_y_name"]

    data_paths = {"data_folder": data_folder, "dataset_X_y_name": dataset_X_y_name}

    X_y = np.load(data_folder + "/" + dataset_X_y_name + ".npz")

    X_with_engine = X_y["X"]
    y = X_y["y"]

    idx_engine_id = X_y["feature_names"].tolist().index("engine_id")

    f_name = X_y["feature_names"].tolist()
    f_name_no_engine_id = f_name.copy()
    f_name_no_engine_id.pop(idx_engine_id)

    X = np.delete(X_with_engine, idx_engine_id, axis=2)  # no engine_id feature

    variable_names = {
        "feature_names_X": f_name_no_engine_id,
        "feature_names_X_with_engine": f_name,
        "y_name": X_y["y_name"].tolist(),
    }

    return X, X_with_engine, y, idx_engine_id, data_paths, variable_names


def train_random_forest(cfg, cfg_rf):

    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # load datasets
    X, X_with_engine, y, data_paths_training = load_tabular_data(cfg, is_training=True)
    X_test, _, y_test, data_paths_test = load_tabular_data(cfg, is_training=False)

    source_trn_X = data_paths_training["data_folder"] + '/' + data_paths_training["dataset_X_name"] + '.csv'
    source_tst_X = data_paths_test["data_folder"] + '/' + data_paths_test["dataset_X_name"] + '.csv'
    source_trn_y = data_paths_training["data_folder"] + '/' + data_paths_training["dataset_y_name"] + '.csv'
    source_tst_y = data_paths_test["data_folder"] + '/' + data_paths_test["dataset_y_name"] + '.csv'
    training_dataset = mlflow.data.from_pandas(df=pd.concat([X,y])) # mlflow log purpose
    test_dataset = mlflow.data.from_pandas(df=pd.concat([X_test,y_test])) # mlflow log purpose


    experiment_name = cfg.get("mlflow_experiment_name", "CMAPSS_Training")
    run_name = cfg.get("mlflow_run_name", "random_forest")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        log_params(flatten_dict({"common": cfg}))
        log_params(flatten_dict({"random_forest": cfg_rf}))

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestRegressor(
                        random_state=cfg["random_state"],
                        n_jobs=cfg_rf["n_jobs"],
                    ),
                ),
            ]
        )

        model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())

        cv = GroupKFold(n_splits=cfg_rf["n_splits_cv"])
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=cfg_rf["hyperparameters"],
            cv=cv,
            scoring=cfg_rf["scoring"],
            n_jobs=cfg_rf["n_jobs"],
            verbose=cfg_rf["verbose"],
        )

        grid_search.fit(X, y, groups=X_with_engine["engine_id"])

        # metrics
        mae_cv_trn = abs(grid_search.best_score_)
        y_test_pred = grid_search.predict(X_test)
        mae_cv_tst = mean_absolute_error(y_test, y_test_pred)

        metrics_trn = {"Training set": data_paths_training, "MAE VAL": mae_cv_trn}
        metrics_tst = {"Test set": data_paths_test, "MAE TST": mae_cv_tst}

        # mlflow logs
        mlflow.log_metric("mae_val", mae_cv_trn)
        mlflow.log_metric("mae_test", mae_cv_tst)
        mlflow.log_param("best_params", json.dumps(grid_search.best_params_))
        mlflow.log_param("dataset_train_rows", X.shape[0])
        mlflow.log_param("dataset_test_rows", X_test.shape[0])
        mlflow.log_param("dataset_feature_count", X.shape[1])
        mlflow.log_param("feature_names", json.dumps(X.columns.tolist()))
        mlflow.log_param("dataset_train_path", json.dumps(data_paths_training))
        mlflow.log_param("dataset_test_path", json.dumps(data_paths_test))
        mlflow.log_input(training_dataset, context="training")
        mlflow.log_input(test_dataset, context="test")
        mlflow.log_artifact(source_trn_X, artifact_path="datasets")
        mlflow.log_artifact(source_tst_X, artifact_path="datasets")
        mlflow.log_artifact(source_trn_y, artifact_path="datasets")
        mlflow.log_artifact(source_tst_y, artifact_path="datasets")
        mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path="model", registered_model_name=cfg["mlflow_registry_model_name"])


def train_lstm(cfg, cfg_lstm):

    import tensorflow as tf
    from sklearn.model_selection import GroupShuffleSplit
    from tensorflow.keras import layers, models

    tf.random.set_seed(cfg["random_state"])

    X, X_with_engine, y, idx_engine_id, data_paths_training, variable_names = load_sequence_data(cfg, is_training=True)
    X_test, _, y_test, _, data_paths_test, _ = load_sequence_data(cfg, is_training=False)

    source_trn = data_paths_training["data_folder"] + '/' + data_paths_training["dataset_X_y_name"] + '.npz'
    source_tst = data_paths_test["data_folder"] + '/' + data_paths_test["dataset_X_y_name"] + '.npz'
    training_dataset = mlflow.data.from_numpy(X, targets=y, source=source_trn) # mlflow log purpose
    test_dataset = mlflow.data.from_numpy(X_test, targets=y_test, source=source_tst) # mlflow log purpose

    experiment_name = cfg.get("mlflow_experiment_name", "CMAPSS_Training")
    run_name = cfg.get("mlflow_run_name", "lstm")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        log_params(flatten_dict({"common": cfg}))
        log_params(flatten_dict({"lstm": cfg_lstm}))

        # GroupShuffleSplit per engine
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg_lstm["train_val_ratio"], random_state=cfg["random_state"])
        train_idx, val_idx = next(gss.split(X, y, groups=X_with_engine[:, idx_engine_id, 0]))

        # split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        norm = layers.Normalization()
        norm.adapt(X_train)

        model = models.Sequential(
            [
                layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
                norm,
                layers.LSTM(cfg_lstm["layers"][0]["units"], return_sequences=True),
                layers.Dropout(cfg_lstm["layers"][0]["dropout"]),
                layers.LSTM(cfg_lstm["layers"][1]["units"], return_sequences=False),
                layers.Dense(1),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg_lstm["learning_rate"]),
            loss=cfg_lstm["loss"],
            metrics=cfg_lstm["metrics"],
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=cfg_lstm["epochs"],
            batch_size=cfg_lstm["batch_size"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor=cfg_lstm["early_stopping"]["monitor"],
                    patience=cfg_lstm["early_stopping"]["patience"],
                    restore_best_weights=cfg_lstm["early_stopping"]["restore_best_weights"],
                )
            ],
            verbose=cfg_lstm["verbose"],
        )

        y_val_pred = model.predict(X_val, verbose=0).flatten()
        y_test_pred = model.predict(X_test, verbose=0).flatten()

        mae_val = mean_absolute_error(y_val[:, -1], y_val_pred)
        mae_test = mean_absolute_error(y_test[:, -1], y_test_pred)

        metrics_trn = {"Training set": data_paths_training, "MAE VAL": mae_val}
        metrics_tst = {"Test set": data_paths_test, "MAE TST": mae_test}

        mlflow.log_metric("mae_val", mae_val)
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_param("dataset_train_sequences", X.shape[0])
        mlflow.log_param("dataset_test_sequences", X_test.shape[0])
        mlflow.log_param("dataset_timesteps", X.shape[1])
        mlflow.log_param("dataset_feature_count", X.shape[2])
        mlflow.log_param("feature_names", json.dumps(variable_names["feature_names_X"]))
        mlflow.log_param("dataset_train_path", json.dumps(data_paths_training))
        mlflow.log_param("dataset_test_path", json.dumps(data_paths_test))
        mlflow.log_input(training_dataset, context="training")
        mlflow.log_input(test_dataset, context="test")
        mlflow.log_artifact(source_trn, artifact_path="datasets")
        mlflow.log_artifact(source_tst, artifact_path="datasets")
        mlflow.log_param("epochs_trained", len(history.history.get("loss", [])))
        mlflow.tensorflow.log_model(model, artifact_path="model", registered_model_name=cfg["mlflow_registry_model_name"])


def main():

    config = load_config(CONFIG_PATH)
    cfg = config["common"]  # common config

    mlflow_server_uri = cfg.get("mlflow_server_uri", "http://127.0.0.1:8080")

    try:
        mlflow.set_tracking_uri(mlflow_server_uri)
        print(f"[MLflow] Correct Server connection at: {mlflow_server_uri}")
    except mlflow.exceptions.MlflowException:
        # fallback lacl
        local_mlflow_dir = os.path.join(PROJECT_ROOT, "mlruns")
        os.makedirs(local_mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(local_mlflow_dir)
        print(f"[MLflow] Server not reachable. Local backend at: {local_mlflow_dir}")

    algorithm = cfg["algorithm"].lower()

    if algorithm == "random_forest":

        if cfg["data_type"] == "sequence":
            raise ValueError("wrong data_type in config, for RF has to be 'tabular'")

        rf_cfg = config["random_forest"]
        train_random_forest(cfg, rf_cfg)

    elif algorithm == "lstm":

        if cfg["data_type"] == "tabular":
            raise ValueError("wrong data_type in config, for LSTM has to be 'sequence'")

        lstm_cfg = config["lstm"]
        train_lstm(cfg, lstm_cfg)

    else:
        raise ValueError("algorithm type not found in config")


if __name__ == "__main__":
    main()
