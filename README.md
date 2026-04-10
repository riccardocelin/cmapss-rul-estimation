# CMAPSS Jet Engine Predictive Maintenance

End-to-end machine learning project for **Remaining Useful Life (RUL)** prediction on NASA CMAPSS turbofan data, including:
- data preprocessing,
- model training and experiment tracking,
- API packaging,
- push on GHCR (with automatic CI workflow via Github Actions)
- kubernetes deployment (local minikube, manual deployment: CD fiesible with minikube set up)

                ┌──────────────────────────┐
                │        TRAINING          │
                │  (experiments, tuning)   │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │     MLFLOW TRACKING      │
                │  (metrics, params, runs) │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │   MODEL REGISTRY         │
                │ (versioned models v1,v2) │
                └──────────┬───────────────┘
                           │
                           │  (after evaluation / validation)
                           ▼
        ┌────────────────────────────────────────────┐
        │   EXPORT SELECTED MODEL (model-vX)         │
        │   → prj_fld/app/model                      │
        └──────────────────────────┬─────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────┐
        │        DOCKER BUILD                        │
        │  (API code + exported model bundled)       |
        └──────────────────────────┬─────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────┐
        │   DOCKER IMAGE                             │
        │   (versioned, reproducible artifact)       │
        └──────────────────────────┬─────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────┐
        │   GITHUB CONTAINER REGISTRY (GHCR)         │
        │   (image storage & distribution)           │
        └──────────────────────────┬─────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────┐
        │   KUBERNETES DEPLOYMENT                    │
        │   (Minikube: LoadBalancer + tunneling)     │
        └──────────────────────────┬─────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────┐
        │   RUNNING API                              │
        │   (inference server)                       │
        └────────────────────────────────────────────┘

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
7. local deployment in kubernetes (minikube)

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

Important note/limitations:
the deployed model is also gitted in prj repo in app/model, this is for sure not a best practice but it is a workaround for this demo project in order to have the model exported from the local registry (mlflow in localhost) and visible by github actions for CI purposes. In real production, the mlflow databases would be remote and the CI pipelines would fetch the model runtime.
Another critical limitation is due to the fact that for this demo project minikube local Kubernetes cluster is used, which cannot be accessed from GitHub Actions. In a production scenario with a cloud-based Kubernetes cluster, the deployment step can be fully automated in the CI/CD pipeline, but in this project the local deployment on K8s require a manual step (for security reasons, a self-hosted runner on github has not been configured)

Notes on deployment workflow:
- model update with ./src/update_model_to_deploy_from_registry.py: this script will update (automatic commit + push) of the latest @champion model on mlflow (config on ./config/check_model_to_deploy.local.yaml)
- PR/push on barnch main will trigger a CI workflow for the automatic build + push on GHCR of the docker image with the model and API.
- After CI workflow, a new image is available on the registry and has to be manually pulled by applying the ./k8s/deployment.yaml manifest. K8s mainifest is versioned on git, and it is configured for pulling the :latest tag image from the registry: in reality, it is suggested to manually change the image tag and explicitly refer to the tag found in GHCR in ordert to correctly manage the k8s features such as rollback or rolling update. These changes should not be tracked (this is a limitation of a standard CD workflow due to the use of minikube for demo purposes).


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

Build image & push on GitHub Container Registry:

1. Build image:
```bash
docker build -t ghcr.io/riccardocelin/cmapss-rul-engine:1.0.1-RFv1 . # docker build -t ghcr.io/<github-owner>/<package-name>:<imageversion>-<model><modelversion> .
```

2. Push image in GHCR:
```bash
docker push ghcr.io/riccardocelin/cmapss-rul-engine:1.0.1-RFv1 # docker push ghcr.io/<github-owner>/<package-name>:<imageversion>-<model><modelversion>
```

3. (Opional) local docker container run (api url: http://127.0.0.1:8000)
```bash
docker run -p 8000:8000 riccardocelin/cmapss-rul-engine:RF-v1 # running the container, exposing port 8000
```

---

## Kubernetes deployment

Minikube workflow for local deployment:
```bash
minikube start -p <cluster-name>
kubectl apply -f k8s/   # (only the first time) Apply all Kubernetes manifests in the folder prj_fld/k8s (deployment.yaml, service.yaml, eventual configmap.yaml) - will be applied to active context
minikube tunnel # enable tunneling (without cloud provider, k8s cannot create a real LoadBalancer: simulated with tunneling in orther to obtain an external IP:port)
kubectl get svc # useful to get the ip:port after the tunneling (//127.0.0.1:80)
python app/test_..._service.py # test model deployed in kubernetes
```

1. Useful commands
```bash
kubectl apply -f k8s/   # Apply all Kubernetes manifests in the folder prj_fld/k8s (deployment.yaml, service.yaml, eventual configmap.yaml)
kubectl apply -f k8s/ --context=<cluster-name> # apply to a specific cluster-name
minikube service <service-name>  # Open the Service in browser
kubectl get all          # List all created resources (after 'kubectl apply -f k8s/')
kubectl get pods         # List all running Pods
kubectl get services     # List Services
kubectl get deployments  # List Deployments
```

2. Pod inspection:
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

3. Debug:
```bash
kubectl describe pod <pod-name> # (example: kubectl describe pod cmapss-rul-api-764b6986-5mt5k)
kubectl rollout status deployment <your-api> # check if rolling update/rollback was succesfully
```

4. Rollback (after deployment update):
```bash
kubectl rollout undo deployment <your-api> # <your-api> is the name of the deployment (field metadata.name in deployment.yaml)
kubectl rollout history deployment <your-api> # k8s read the replicaset and provides history
```

Conceptual notes:
- Deployment = guarantees N active Pods
- Pod = containers wrapper
- Service = routing through Pod via label
- ConfigMap = external config
the Service does not see the containters itself, but it does see the pod (via label match)

In ML systesms, memory dimensioning is critical because each replica loads the model inside the RAM, too low limits can lead to outofmemory killing and system instability.
Kubernetes thinks about CPU resources in cores or millicpu: CPU:100m means the the limit is 100 milli cpu, the the limit is 10% of a CPU.


Understanding Ports in Kubernetes (Container → Pod → Service → External Access)
When deploying an application on Kubernetes, it’s important to clearly understand how networking and ports work across different layers. A common source of confusion is assuming that the port your application uses internally is the same one you should use externally — this is not the case.

Architecture Overview:
The request flow looks like this:
Client → Node → Service → Pod → Container
Each layer has a specific role and may expose different ports.

1. Container Level
Inside the container, your application runs on a specific port.
Example (FastAPI):
uvicorn.run(app, host="0.0.0.0", port=8000)
This means:
The application listens on port 8000
This port is only accessible inside the container (and pod)

2. Pod Level
A Pod is a wrapper around one or more containers.
The Pod inherits the container port
No additional port mapping happens here
So:
Pod port = 8000

3. Service Level
A Service exposes your Pods and enables communication within the cluster (and optionally outside).
Example configuration:
ports:
  - port: 80
    targetPort: 8000
    nodePort: 30007

Meaning of each field:
targetPort: 8000
The port on the container (your application)
port: 80
The internal Service port (used inside the Kubernetes cluster)
nodePort: 30007
The port exposed externally on the node (used by clients)

4. Full Request Flow
When you send a request:
curl http://<NODE_IP>:30007/predict
The request flows as follows:
Client
  ↓ (port 30007)
Node (Minikube)
  ↓
Service (port 80)
  ↓
Pod (port 8000)
  ↓
Container (FastAPI app)

Key Rule: Always use the NodePort (external port) to access your application from outside the cluster.



In Kubernetes, a NodePort Service:
exposes the service on a specific port on each node in the cluster
Diagram:
Client → <NodeIP>:<NodePort> → Service → Pod
Example:
192.168.49.2:30007
Meaning:
Each node in the cluster listens on port 30007
Traffic is forwarded to the target pods by the Service


In Kubernetes, a LoadBalancer Service:
tells Kubernetes:
“I want this service to be accessible from the outside via a public IP address”
Logical flow:
Client → LoadBalancer (public IP) → Node → Service → Pod
The service is delegated to a cloud service, in minikube there is no cloud provider, so the LoadBalancer -> EXTERNAL-IP: pending

The Minikube tunnel is a local process that:
simulates a real LoadBalancer

bash: Minikube tunnel
- Minikube: creates a network route on your host
- assigns an external IP address to the Service
- forwards traffic from your host → cluster

Actual flow with tunnel:
Client (my Mac)
   ↓
External IP (assigned by Minikube)
   ↓
Tunnel (local process)
   ↓
Service → Pod

Summary
NodePort:
Client → NodeIP:port
LoadBalancer:
Client → public and stable IP address


---

## Azure Container Apps deployment (example flow - not used, now K8s deployment)

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