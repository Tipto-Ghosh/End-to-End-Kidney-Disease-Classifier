import os 
from dotenv import load_dotenv 
from pathlib import Path


load_dotenv()

CONFIG_FILE_PATH = "config/config.yaml"
PARAMS_FILE_PATH = "config/params.yaml"

ARTIFACT_DIR : str = "artifacts"

DATA_SOURCE_URL  = os.getenv("data_source_url")
DATA_INGESTION_DIR_NAME : str = "data_ingestion"
DATA_INGESTION_LOCAL_FILE : str = "data.zip"
DATA_INGESTION_UNZIP_DIR : str = "data_ingestion"


BASE_MODEL_ROOT_DIR = "prepare_base_model"
BASE_MODEL_FILE_PATH = "base_model.pt"
UPDATED_BASE_MODEL_FILE_PATH = "base_model_updated.pth"
UPDATED_MODEL_SUMMARY_FILE_PATH = "_architecture_summary.txt"