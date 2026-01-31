import os 
from dotenv import load_dotenv 
from pathlib import Path


load_dotenv()

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")

ARTIFACT_DIR : str = "artifacts"

DATA_SOURCE_URL  = os.getenv("data_source_url")
DATA_INGESTION_DIR_NAME : str = "data_ingestion"
DATA_INGESTION_LOCAL_FILE : str = "data.zip"
DATA_INGESTION_UNZIP_DIR : str = "data_ingestion"

