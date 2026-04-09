#!/bin/bash

# This script builds the Docker image for the RUL inference service. It should be run from the root of the project.

set -e  # interrupt the script if any command fails

IMAGE_NAME="ghcr.io/riccardocelin/cmapss-rul-engine" # has to be consistent with the image name in k8s/deployment.yaml

GIT_SHA=$(git rev-parse --short HEAD)

MODEL_NAME_VERSION=$(python ./cicd/get_model_info_for_cicd.py)

TAG="${GIT_SHA}-$MODEL_NAME_VERSION"

docker build -t $IMAGE_NAME:$TAG .

echo $IMAGE_NAME:$TAG