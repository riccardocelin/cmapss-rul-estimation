#!/bin/bash

# CI script: This script builds the Docker image for the RUL inference service. It should be run from the root of the project.

set -e  # interrupt the script if any command fails

IMAGE_NAME="ghcr.io/riccardocelin/cmapss-rul-engine" # has to be consistent with the image name in k8s/deployment.yaml
TAG="1.0.3-RFv1" # ???? has to be automatically retrieved from model artifact and previous image (HOW?????)

echo ">> CI: Building Docker image..."

docker build -t $IMAGE_NAME:$TAG .
docker push $IMAGE_NAME:$TAG

echo "<< CI: Image built and pushed on GHCR successfully: $IMAGE_NAME:$TAG"