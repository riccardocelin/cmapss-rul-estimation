# script to test the deployed RUL inference service by sending requests with test data and printing the predictions and model info.
# works both locally (local container or kubernetes/minikube) and on cloud (e.g., when deployed on AWS/GCP/Azure) by updating the 'base_url' in the config file.

import os
import json
import requests
import numpy as np
import pandas as pd

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "configs",
    "service_test_config.local.json",
)


def _load_service_test_config(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing config file: {config_path}. "
            "Copy configs/service_test_config.template.json to configs/service_test_config.local.json and update it."
        )

    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


_SERVICE_TEST_CONFIG = _load_service_test_config()

base_url = _SERVICE_TEST_CONFIG["base_url"]
endpoint = _SERVICE_TEST_CONFIG["endpoint"]
data_folder = _SERVICE_TEST_CONFIG["data_folder"]
sequence_model = _SERVICE_TEST_CONFIG["sequence_model"]  # Sequence model: True -> test for LSTM models (sequence), model: False -> test for RF models (tabular)
loop_mod = _SERVICE_TEST_CONFIG["loop"] 


def predict(url="http://127.0.0.1:8000/predict", data=None, verbose=True, timeout=50):
    """
    Make a POST request to get the prediction from the RUL model served via FastAPI.

    Args:
        url: URL that the request is sent to.
        data: model input
        verbose: print diagnostic information.
        timeout: request timeout in seconds.

    Returns:
        dict: parsed JSON response from the server.
    """

    if data is None:
        raise ValueError("'data' cannot be None.")

    # Accept numpy arrays directly and convert to JSON-serializable structure.
    #sequences = data.tolist() if isinstance(data, np.ndarray) else data
    sequences = None
    if sequence_model:
        sequences = data.tolist()
    else:
        sequences = data.values.tolist()

    payload = {"inputs": sequences}

    response = requests.post(url, json=payload, timeout=timeout)

    if verbose:
        print(f"POST {url} -> {response.status_code}")
        if response.ok:
            print("Prediction succeeded.")
        else:
            print(f"Request failed: {response.text}")

    # Raise a useful error if the API returned non-200.
    response.raise_for_status()
    return response.json()

def load_test_input(data_fullfile, sequence_flag=True):

    X = None

    # load test data for api prediction
    if sequence_flag:
        X_y = np.load(data_fullfile)

        X_with_engine = X_y["X"]
        y = X_y["y"]

        idx_engine_id = X_y["feature_names"].tolist().index("engine_id")

        f_name = X_y["feature_names"].tolist()
        f_name_no_engine_id = f_name.copy()
        f_name_no_engine_id.pop(idx_engine_id)

        X = np.delete(X_with_engine, idx_engine_id, axis=2)  # no engine_id feature

    else: # tabular data
        X_with_engine = pd.read_csv(data_fullfile)
        X = X_with_engine.loc[:, X_with_engine.columns != "engine_id"]

    return X


def main():

    url = base_url + endpoint
    actual_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.dirname(actual_dir) + "/" + data_folder

    X = load_test_input(data_dir, sequence_model)

    # test server connection
    test_response = requests.get(base_url + '/')
    print(f"Test response: {test_response}")

    while True:
        response = predict(url, X)
        model_info = requests.get(base_url + '/model_info').json()
        print(f"Predicted RUL:  {response['predictions']}\n")
        print(f"Model name:     {model_info['model_name']}\n")
        print(f"Model version:  {model_info['model_version']}\n")

        if not loop_mod:
            break

if __name__ == "__main__":
    main()
