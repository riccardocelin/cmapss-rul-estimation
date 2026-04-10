#!/bin/bash

# Deployment script: This script deploys the RUL inference service to Kubernetes. It should be run from the root of the project.
# Limitiation: this script should be manually triggered after the CI pipeline, this is a limitation
# due to the fact that for this demo project minikube local Kubernetes cluster is used, which cannot be accessed from GitHub Actions.
# In a production scenario with a cloud-based Kubernetes cluster, the deployment step can be fully automated in the CI/CD pipeline.

set -e

echo ">> Deploying to Kubernetes using local k8s manifest config..."

kubectl apply -f k8s/ deployment.yaml

echo "<< Deployment applied successfully"