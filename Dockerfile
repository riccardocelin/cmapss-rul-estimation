# syntax=docker/dockerfile:1

# Dockerfile defines how to build the docker image

FROM python:3.11.14-slim

WORKDIR /app

# install requirements from app
COPY app/requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# install requirements from mlflow artifacts
COPY app/model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

LABEL org.opencontainers.image.source=https://github.com/riccardocelin/cmapss-rul-estimation
