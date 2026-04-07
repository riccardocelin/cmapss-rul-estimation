from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import List, Any
import yaml
import os
import mlflow.pyfunc

# NOTE: loading a model exported from mlflow local server to app/model is a workaround to avoid using local mlflow uri due to the fact that this demo project aim to deploy the model on cloud, in this case the choice of downloading the model in app/models is a tradeoff choice in order to maintain simplicity for the deployment while using mlflow model registry tool in localhost (local training)

# Load the model version currently under the 'champion' alias
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "model")
MODEL_FILE_YAML = MODEL_PATH + "/registered_model_meta"  # this is the model file that will be used for deployment, it will be updated if the model in MLflow registry is updated

model_file_content = None
model_name = "NaN"
model_version = "NaN"
if os.path.exists(MODEL_FILE_YAML):
    # read yaml file in model_file_yaml to get the actual model name and version
    with open(MODEL_FILE_YAML, "r") as f:
        model_file_content = yaml.safe_load(f)

    model_name = str(model_file_content["model_name"])
    model_version = str(model_file_content["model_version"])


app = FastAPI()

model = None
model_ready = False

class DataInstance(BaseModel):
    inputs: List[Any]

@app.lifespan("startup")
def load_model_on_startup():
    global model, model_ready
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    model_ready = True

@app.get("/")
def read_root():
    return{"health_status": "ok"}

@app.get("/health/live")
def liveness():
    return {"status": "alive"}

@app.get("/health/ready")
def readiness():
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}

@app.get("/model_info")
def model_info():
    return {
        "model_name": model_name,
        "model_version": model_version
    }

@app.post("/predict")
def predict(input_data: DataInstance):
    x = input_data.inputs
    preds = model.predict(x)
    return {"predictions": preds.tolist()}