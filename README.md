# CMAPSS RUL Estimation

End-to-end machine learning project for **Remaining Useful Life (RUL)** estimation on the NASA CMAPSS turbofan degradation dataset.

The repository includes:
- data preprocessing for tabular and sequence datasets,
- training pipelines for Random Forest and LSTM models,
- experiment tracking and model registry with MLflow,
- a FastAPI inference service,
- Docker packaging and image publishing to GHCR,
- Kubernetes manifests for local Minikube deployment.

---

## Project goal

Given multivariate time-series from turbofan engines, estimate the number of cycles remaining before failure (**RUL**).

This project demonstrates an end-to-end MLOps flow:
1. raw data ingestion and feature engineering,
2. model training and evaluation,
3. experiment tracking in MLflow,
4. model promotion/export for serving,
5. API inference with FastAPI,
6. containerization with Docker,
7. deployment to local Kubernetes (Minikube).

---

## Repository structure

### Core training and data pipeline
- `src/data_preprocess/dataset_generation.py`  
  Builds processed datasets from raw CMAPSS files. Supports tabular CSV output and sequence NPZ output.
- `src/data_preprocess/data_generation_fcn.py`  
  Utility functions for loading CMAPSS files, RUL generation, NaN handling, and feature filtering.
- `src/training.py`  
  Trains either Random Forest (`algorithm=random_forest`) or LSTM (`algorithm=lstm`) and logs runs to MLflow.

### Model serving and deployment
- `app/api.py`  
  FastAPI app exposing inference and health endpoints. Loads model artifacts from `app/model/`.
- `src/update_model_to_deploy_from_registry.py`  
  Pulls the `@champion` model from MLflow Registry into `app/model/` and can auto-commit/push changes.
- `src/test_deployed_service.py`  
  Sends test requests to local/cloud service endpoints and prints predictions/model metadata.

### Infrastructure and CI/CD support
- `Dockerfile`  
  Builds the API image (FastAPI + model artifact).
- `k8s/deployment.yaml`, `k8s/service.yaml`  
  Kubernetes deployment/service manifests.
- `.github/workflows/ci.yaml`
  GitHub Actions workflow that runs on `pull_request` and `push` to `main` (for `app/**` changes).
- `cicd/build.sh`, `cicd/push.sh`, `cicd/deploy.sh`  
  Helper scripts for image build/push and Kubernetes deployment.
- `cicd/get_model_info_for_cicd.py`  
  Extracts model name/version from `app/model/registered_model_meta` for image tagging.

### Config templates
- `configs/dataset_generation_config.template.json`
- `configs/training_config.template.json`
- `configs/check_model_to_deploy.template.yaml`
- `configs/service_test_config.template.json`

### Example notebooks
- `example_training_ML.ipynb`
- `example_training_TF.ipynb`

---

## Important limitations (intentional for demo scope)

- The deployed model artifact is versioned in-repo under `app/model/`. This is convenient for this demo but is **not** ideal for production.
- CI is visible in GitHub Actions, but CD is intentionally **not fully visible/automated** in GitHub Actions for this demo.
- Kubernetes CD is manual in this project because the target cluster is local Minikube (not reachable from GitHub-hosted runners).
- Deployment is performed manually by updating `k8s/deployment.yaml` with the real image tag published in GHCR (replace `latest` with the target tag), then applying the manifest with `kubectl apply -f k8s/deployment.yaml`.
- In production, MLflow services and artifact storage should be remote/shared, and deployment should run in a fully automated CD pipeline.

---

## Setup

### 1) Create Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Add raw CMAPSS dataset files

Place data files in:

- `data/CMAPSSData/train_FD00x.txt`
- `data/CMAPSSData/test_FD00x.txt`
- `data/CMAPSSData/RUL_FD00x.txt`

> Raw CMAPSS files are not committed in this repository.

---

## 1) Generate processed datasets

Create local config from template:

```bash
cp configs/dataset_generation_config.template.json configs/dataset_generation_config.local.json
```

Edit the local config (for example: `is_sequence_modeling`, `sequence_len`, `max_rul`, `dataset_version`), then run:

```bash
python src/data_preprocess/dataset_generation.py
```

Output paths:
- tabular mode: `data/processed/tabular/<dataset_version>/`
- sequence mode: `data/processed/sequence/<dataset_version>/`

## 2) Train model and log to MLflow

Start MLflow tracking server:

```bash
mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri ./mlruns --artifacts-destination ./mlruns
```

Create local training config:

```bash
cp configs/training_config.template.json configs/training_config.local.json
```

Update at least:
- `common.algorithm`: `"random_forest"` or `"lstm"`
- `common.data_type`: `"tabular"` (RF) or `"sequence"` (LSTM)
- training/test dataset names and version fields

Run training:

```bash
python src/training.py
```

MLflow UI:
- `http://127.0.0.1:8080`

## 3) Export/update deployment model from MLflow Registry

Create local config:

```bash
cp configs/check_model_to_deploy.template.yaml configs/check_model_to_deploy.local.yaml
```

Set:
- `mlflow.tracking_uri`
- `model.registry_name`
- `model.alias` (for example `champion`)
- `export.output_dir` (typically `app/model`)

Run:

```bash
python src/update_model_to_deploy_from_registry.py
```

This script updates `app/model/` with the selected model artifact and includes an internal auto git commit/push routine when changes are detected.

## 4) Run API locally

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Main endpoints:
- `GET /`
- `GET /health/live`
- `GET /health/ready`
- `GET /model_info`
- `POST /predict`

## 5) Test deployed service

Create local config:

```bash
cp configs/service_test_config.template.json configs/service_test_config.local.json
```

Run test client:

```bash
python src/test_deployed_service.py
```

---

## Docker

Build image manually:

```bash
docker build -t ghcr.io/<github-owner>/<package-name>:<tag> .
```

Push image:

```bash
docker push ghcr.io/<github-owner>/<package-name>:<tag>
```

Run locally:

```bash
docker run -p 8000:8000 ghcr.io/<github-owner>/<package-name>:<tag>
```

### CI helper scripts

From repo root:

```bash
bash cicd/build.sh
bash cicd/push.sh
```

Both scripts derive image tags from `<git-sha>-<model_name>v<model_version>`.

### GitHub Actions CI behavior

- The workflow is defined in `.github/workflows/ci.yaml`.
- CI is triggered on:
  - `pull_request` targeting branch `main`
  - `push` to branch `main`
- The workflow currently filters on `app/**` path changes.
- On `push` to `main`, the image push step runs (GHCR push).

### CD note (demo limitation)

This repository uses a **local Minikube cluster** for deployment demos. Because that cluster is local, GitHub-hosted runners cannot access it; therefore CD is manual by design in this project.

Manual deployment flow:
1. Identify the image tag pushed to GHCR (from CI output or by listing package versions in GHCR).
2. Edit `k8s/deployment.yaml` and replace the image tag `latest` with that real GHCR tag.
3. Apply the manifest manually:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```
4. Verify rollout:
   ```bash
   kubectl rollout status deployment/cmapss-rul-api
   ```

> This manual CD process is an explicit limitation of the current demo structure, not the recommended production approach.

---

## Kubernetes (Minikube)

Start cluster and deploy manifests:

```bash
minikube start -p <cluster-name>
kubectl apply -f k8s/
minikube tunnel
kubectl get svc
```

Useful commands:

```bash
kubectl get all
kubectl get pods
kubectl get deployments
kubectl get services
kubectl describe pod <pod-name>
kubectl logs <pod-name>
kubectl rollout status deployment <deployment-name>
kubectl rollout history deployment <deployment-name>
kubectl rollout undo deployment <deployment-name>
```

Notes:
- `k8s/service.yaml` is configured as `LoadBalancer`; in Minikube, `minikube tunnel` provides external access.
- Use health endpoints (`/health/live`, `/health/ready`) to validate pod status.

---

## Notes

- Local `*.local.json` and `*.local.yaml` files are expected for personal settings and should not contain secrets in committed form.
- This project is designed as a practical MLOps portfolio/demo project rather than a production-hardened platform.
