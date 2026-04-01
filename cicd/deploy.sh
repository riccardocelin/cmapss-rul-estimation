#!/bin/bash

# CD script: This script deploys the RUL inference service to Kubernetes. It should be run from the root of the project.

set -e

echo ">> CD: Deploying to Kubernetes..."

kubectl apply -f k8s/ deployment.yaml

echo "<< CD: Deployment applied successfully"