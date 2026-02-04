import os 
from dotenv import load_dotenv 
from pathlib import Path


load_dotenv()

CONFIG_FILE_PATH = "config/config.yaml"
PARAMS_FILE_PATH = "config/params.yaml"

# Artifact directory
ARTIFACT_DIR : str = "artifacts"

# Data Ingestion Constants
DATA_SOURCE_URL = os.getenv("data_source_url")
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_LOCAL_FILE: str = "data.zip"
DATA_INGESTION_UNZIP_DIR: str = "extracted_data"
DATA_INGESTION_TRAIN_DIR: str = "train"
DATA_INGESTION_TEST_DIR: str = "test"
DATA_INGESTION_VALIDATION_DIR: str = "validation"
DATA_INGESTION_TRAIN_SIZE: float = 0.7
DATA_INGESTION_TEST_SIZE: float = 0.15
DATA_INGESTION_VALIDATION_SIZE: float = 0.15

# Prepare Base Model Constants
PREPARE_BASE_MODEL_DIR_NAME: str = "prepare_base_model"
BASE_MODEL_DIR: str = "base_model"
UPDATED_MODEL_DIR: str = "updated_model"
BASE_MODEL_FILE_NAME: str = "base_model.pth"
UPDATED_MODEL_FILE_NAME: str = "updated_model.pth"
MODEL_SUMMARY_FILE_NAME: str = "model_summary.txt"

