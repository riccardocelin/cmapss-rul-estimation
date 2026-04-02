# NOTE: this is a workaround to avoid using local mlflow uri due to the fact that this demo project aim to deploy the model on cloud
import mlflow
from mlflow.tracking import MlflowClient
import shutil
import yaml
import os
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(PROJECT_ROOT) + "/configs/check_model_to_deploy.local.yaml"  # remember to create a new file copy from configs/download_model_to_deploy.template.yaml
MODEL_FILE_YAML = str(PROJECT_ROOT) + "/model_deployed.yaml"  # this is the model file that will be used for deployment, it will be updated if the model in MLflow registry is updated

def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    tracking_uri = config["mlflow"]["tracking_uri"]
    model_registry_name = config["model"]["registry_name"]
    alias = config["model"]["alias"]
    output_dir = config["export"]["output_dir"]

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(
        name=model_registry_name,
        alias=alias
    )
    champion_model = {
        "registry_name": str(model_registry_name),
        "version": str(model_version.version)
    }

    updated = check_and_update_model_file(MODEL_FILE_YAML, champion_model)

    if updated:

        print(f"Model file in {MODEL_FILE_YAML} updated successfully. Copying to {output_dir}...")

        model_uri = f"models:/{model_registry_name}@{alias}"
        model_artifact = mlflow.artifacts.download_artifacts(model_uri)

        # delete the old model file in the output directory before copying the new one
        shutil.rmtree(output_dir, ignore_errors=True)

        shutil.copytree(model_artifact, output_dir, dirs_exist_ok=True)

        # manual add of 'uvicorn' requirements when copying from mlflow artifacts
        with open(output_dir+'/requirements.txt', 'a') as f:
            f.write('\n')
            f.write('uvicorn\n')
            f.write('fastapi\n')

        print("Model file copied successfully. Auto Git commit and push the changes...")
        git_commit_and_push()
        print("Auto Git commit and push completed (if configured, CI/CD pipeline will be triggered to build and deploy the updated model).")

    else:
        print("Model file is already up to date. No action needed.")


def check_and_update_model_file(model_file_yaml, champion_model):
    """
    Check if the model file in the output directory is up to date with the latest model from MLflow. If not, update it.

    Returns:
        bool: True if the model file was updated, False if it was already up to date.
    """

    champion_name = champion_model["registry_name"]
    champion_version = champion_model["version"]

    model_file_content = None

    if not os.path.exists(model_file_yaml):
        raise FileNotFoundError(f"Model file not found in MLflow artifacts at {model_file_yaml}.")

    # read yaml file in model_file_yaml to get the actual model name and version
    with open(model_file_yaml, "r") as f:
        model_file_content = yaml.safe_load(f)
    
    actual_name = model_file_content["model"]["registry_name"]
    actual_version = model_file_content["model"]["version"]

    if actual_name != champion_name or actual_version != champion_version:
        print(f"Model file actual content: name={actual_name}, version={actual_version}")
        print(f"Champion model from MLflow registry: name={champion_name}, version={champion_version}")
        print("Model file is outdated. It will be updated.")

        # update the model file with the champion model info
        model_file_content["previous"]["registry_name"]         = str(model_file_content["model"]["registry_name"])
        model_file_content["previous"]["version"]               = str(model_file_content["model"]["version"])
        model_file_content["previous"]["datetime_file_edit"]    = str(model_file_content["model"]["datetime_file_edit"])

        model_file_content["model"]["registry_name"]        = champion_name
        model_file_content["model"]["version"]              = champion_version
        model_file_content["model"]["datetime_file_edit"]   = str(datetime.datetime.now().isoformat())

        with open(model_file_yaml, "w") as f:
            yaml.safe_dump(model_file_content, f)

        return True
    else:
        print("Model file is already up to date.")
        return False
    

def git_commit_and_push():
    """
    Auto commit and push the changes to Git. This will trigger the CI/CD pipeline to build and deploy the updated model.
    Note: this function assumes that Git is configured and the user has the necessary permissions to push to the repository.
    """

    os.system('git add .')
    os.system('git commit -m "automatic commit: update model files for deployment"')
    os.system('git push')

if __name__ == "__main__":
    main()