from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE_YAML = str(PROJECT_ROOT) + "/app/model/registered_model_meta"

with open(MODEL_FILE_YAML) as f:
    config = yaml.safe_load(f)

model_name = str(config["model_name"])
model_version = str(config["model_version"])

print(f"{model_name}v{model_version}")