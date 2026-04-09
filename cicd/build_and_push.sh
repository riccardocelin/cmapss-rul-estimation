#!/bin/bash

# This script builds and pushes the Docker image for the RUL inference service. It should be run from the root of the project.

IMAGE_NAME_TAG=$(bash ./cicd/build.sh)

IFS=':' read -r IMAGE_NAME TAG <<< "$IMAGE_NAME_TAG"

docker push $IMAGE_NAME:$TAG