# NOTE: this is a workaround to avoid using local mlflow uri due to the fact that this demo project aim to deploy the model on cloud
import mlflow
from mlflow.tracking import MlflowClient
import shutil
import yaml
import os
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(PROJECT_ROOT) + "/configs/check_model_to_deploy.local.yaml"  # remember to create a new file copy from configs/download_model_to_deploy.template.yaml
MODEL_FLDR = str(PROJECT_ROOT) + "/app/model"  # this is the model folder that will be used for deployment, it will be updated if the model in MLflow registry is updated
MODEL_FILE_YAML = MODEL_FLDR + "/registered_model_meta"  # this is the model file that will be used for deployment, it will be updated if the model in MLflow registry is updated


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

        print(f"Deployed model updated is out to date with the champion model in MLflow registry. Updating the model file and copying the new model artifact to the output directory for deployment...")

        model_uri = f"models:/{model_registry_name}@{alias}"
        model_artifact = mlflow.artifacts.download_artifacts(model_uri)

        # delete the old model file in the output directory before copying the new one
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Old model file in the output directory {output_dir} has been deleted.")

        shutil.copytree(model_artifact, output_dir, dirs_exist_ok=True)
        print(f"New model artifact downloaded from MLflow registry and copied to the output directory {output_dir} successfully.")

        print("Auto Git commit and push the changes.")
        git_commit_and_push(MODEL_FLDR)
        print("Auto Git commit and push completed.")

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
        print(f"Model file {model_file_yaml} does not exist. Model file will be created with the champion model info retrieved from mlflow.")
        return True

    # read yaml file in model_file_yaml to get the actual model name and version
    with open(model_file_yaml, "r") as f:
        model_file_content = yaml.safe_load(f)
    
    actual_name = model_file_content["model_name"]
    actual_version = model_file_content["model_version"]

    if actual_name != champion_name or actual_version != champion_version:
        print(f"Model file actual content: name={actual_name}, version={actual_version}")
        print(f"Champion model from MLflow registry: name={champion_name}, version={champion_version}")
        print("Model file is outdated. It will be updated for new deployment.")
        return True
    else:
        print("Model file is already up to date.")
        return False


def git_commit_and_push(folder_to_git="./app/model"):
    """
    Auto commit and push the changes to Git. This will trigger the CI/CD pipeline to build and deploy the updated model.
    Note: this function assumes that Git is configured and the user has the necessary permissions to push to the repository.
    """

    def run_cmd(cmd):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
        return result.stdout

    run_cmd(f'git add {folder_to_git}')
    
    try:
        run_cmd('git commit -m "automatic model commit: update model files for CI"')
    except RuntimeError as e:
        if "nothing to commit" in str(e):
            print("No changes to commit")
            return
        else:
            raise

    run_cmd('git push')

if __name__ == "__main__":
    main()