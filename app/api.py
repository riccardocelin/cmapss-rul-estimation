from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import List, Any
import mlflow.pyfunc

# NOTE: loading a model exported from mlflow local server to app/model is a workaround to avoid using local mlflow uri due to the fact that this demo project aim to deploy the model on cloud, in this case the choice of downloading the model in app/models is a tradeoff choice in order to maintain simplicity for the deployment while using mlflow model registry tool in localhost (local training)

# Load the model version currently under the 'champion' alias
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "model")

model = mlflow.pyfunc.load_model(MODEL_PATH)

app = FastAPI()

class DataInstance(BaseModel):
    inputs: List[Any]

@app.get("/")
def read_root():
    return{"health_status": "ok"}

@app.post("/predict")
def predict(input_data: DataInstance):
    x = input_data.inputs
    preds = model.predict(x)
    return {"predictions": preds.tolist()}