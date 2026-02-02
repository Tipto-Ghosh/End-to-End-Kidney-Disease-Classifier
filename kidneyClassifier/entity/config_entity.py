from kidneyClassifier.constants import *
from dataclasses import dataclass
from pathlib import Path
import os 

@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir: str = os.path.join(ARTIFACT_DIR,DATA_INGESTION_DIR_NAME)
    source_url: str = DATA_SOURCE_URL
    local_data_file: str = os.path.join(ARTIFACT_DIR , DATA_INGESTION_DIR_NAME,DATA_INGESTION_LOCAL_FILE)
    unzip_dir: str = root_dir


@dataclass(frozen = True)
class PrepareBaseModelConfig:
    root_dir: str = os.path.join(ARTIFACT_DIR , BASE_MODEL_ROOT_DIR)
    base_model_file_path: str = os.path.join(
        ARTIFACT_DIR, BASE_MODEL_ROOT_DIR, BASE_MODEL_FILE_PATH
    )
    updated_base_model_path: str = os.path.join(
        ARTIFACT_DIR, BASE_MODEL_ROOT_DIR, UPDATED_BASE_MODEL_FILE_PATH
    )
    updated_model_architecture_summary_file_path: str = os.path.join(
        ARTIFACT_DIR,UPDATED_MODEL_SUMMARY_FILE_PATH
    )