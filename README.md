# CMAPSS Jet Engine Predictive Maintenance

End-to-end machine learning project for **Remaining Useful Life (RUL)** prediction on NASA CMAPSS turbofan data, including:
- data preprocessing,
- model training and experiment tracking,
- API packaging,
- and cloud deployment on **Azure Container Apps**.

---

## Project goal

The goal is to estimate how many cycles remain before each engine failure (**RUL**) from multivariate sensor and operating-condition time-series.

This repository demonstrates a full ML lifecycle:
1. raw data ingestion and feature preparation,
2. training with either classical ML (Random Forest) or Deep Learning (LSTM),
3. experiment tracking with MLflow,
4. model export for deployment,
5. online inference through FastAPI,
6. containerization with Docker,
7. cloud deployment to Azure Container Apps.

---

## Repository overview

- `src/data_preprocess/dataset_generation.py`
  Builds processed train/test datasets from raw CMAPSS files (tabular or sequence format).
- `src/data_preprocess/data_generation_fcn.py`
  Utility functions: loading data, RUL generation, missing-data handling, and constant-feature removal.
- `src/training.py`
  Trains and evaluates either `random_forest` or `lstm`, then logs metrics/artifacts to MLflow.
- `app/api.py`
  FastAPI service that loads a model from `app/model/` and serves predictions.
- `app/download_model_to_deploy.py`
  Downloads a model artifact from MLflow Model Registry into `app/model/` for deployment.
- `app/test_local_or_cloud_service.py`
  Script to test local or cloud API inference.
- `configs/*.template.*`
  Templates for preprocessing, training, model export, and API testing.
- `example_training_ML.ipynb`, `example_training_TF.ipynb`
  Demonstration notebooks for RF and LSTM workflows.

---

## Setup

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Add raw CMAPSS data

Place files under:

- `data/CMAPSSData/train_FD00x.txt`
- `data/CMAPSSData/test_FD00x.txt`
- `data/CMAPSSData/RUL_FD00x.txt`

> Raw CMAPSS files are not included in this repository.

---

## End-to-end usage

### Step A — Generate processed datasets

1. Create local preprocessing config:

```bash
cp configs/dataset_generation_config.template.json configs/dataset_generation_config.local.json
```

2. Edit `configs/dataset_generation_config.local.json` based on your paths and preferences (e.g., `is_sequence_modeling`, `sequence_len`, `max_rul`).

3. Run preprocessing:

```bash
python src/data_preprocess/dataset_generation.py
```

Output is saved to:
- `data/processed/tabular/<dataset_version>/...` for tabular mode,
- `data/processed/sequence/<dataset_version>/...` for sequence mode.

### Step B — Train model and track experiments (MLflow)

1. Start MLflow server (port 8080):

```bash
mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri ./mlruns --artifacts-destination ./mlruns
```

2. Create local training config:

```bash
cp configs/training_config.template.json configs/training_config.local.json
```

3. Edit training config:
- `common.algorithm`: `"random_forest"` or `"lstm"`
- `common.data_type`: `"tabular"` (RF) or `"sequence"` (LSTM)
- dataset names under `training` and `test`

4. Run training:

```bash
python src/training.py
```

MLflow UI:
- `http://127.0.0.1:8080`

### Step C — Export selected model for API deployment

1. Create export config:

```bash
cp configs/download_model_to_deploy.template.yaml configs/download_model_to_deploy.local.yaml
```

2. Set tracking URI, registered model name, alias (e.g., `champion`), and output directory.

3. Download model artifact for serving:

```bash
python app/download_model_to_deploy.py
```

The model is copied to `app/model/`.

### Step D — Run API locally

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /` health check
- `POST /predict` predictions

Test API (local or cloud):

```bash
cp configs/service_test_config.template.json configs/service_test_config.local.json
python app/test_local_or_cloud_service.py
```

---

## Docker

Build image & run:

```bash
docker build -t rul-engine
```

```bash
docker run -p 8000:8000 rul-engine
```

Local API URL:
- `http://127.0.0.1:8000`

---

## Azure Container Apps deployment (example flow)

Below is an example CLI flow used to deploy the containerized API.

```bash
# 1) Resource group
az group create --name rg-ml-app --location westeurope

# 2) Azure Container Registry
az acr create --name mlopsdemoregistry --resource-group rg-ml-app --sku Standard
az acr login --name mlopsdemoregistry

# 3) Tag + push image
docker tag cmapssjetenginerul-server mlopsdemoregistry.azurecr.io/cmapssrul:latest
docker push mlopsdemoregistry.azurecr.io/cmapssrul:latest

# 4) Container Apps environment
az containerapp env create --name env-ml-app --resource-group rg-ml-app --location northeurope

# 5) Enable ACR admin (simple demo setup)
az acr update -n mlopsdemoregistry --admin-enabled true

# 6) Deploy Container App
az containerapp create \
  --name ml-api \
  --resource-group rg-ml-app \
  --environment env-ml-app \
  --image mlopsdemoregistry.azurecr.io/cmapssrul:latest \
  --registry-server mlopsdemoregistry.azurecr.io \
  --registry-username $(az acr credential show --name mlopsdemoregistry --query username -o tsv) \
  --registry-password $(az acr credential show --name mlopsdemoregistry --query passwords[0].value -o tsv) \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 1 \
  --max-replicas 3

# 7) Retrieve public endpoint
az containerapp show --name ml-api --resource-group rg-ml-app --query properties.configuration.ingress.fqdn --output tsv
```

> Current example endpoint from this project:
> `https://ml-api.agreeableground-e95cbc53.northeurope.azurecontainerapps.io`

---

## Notes

- This project is intentionally structured as a practical portfolio demo of the complete ML lifecycle.
- Local `.local.json` / `.local.yaml` config files are expected and should not be committed with secrets.
