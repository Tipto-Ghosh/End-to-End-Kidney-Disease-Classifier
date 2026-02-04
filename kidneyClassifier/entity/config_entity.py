import os
from dataclasses import dataclass
from pathlib import Path
from kidneyClassifier.constants import *


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
    source_url: str = DATA_SOURCE_URL
    local_data_file: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_LOCAL_FILE)
    unzip_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_UNZIP_DIR)
    
    # Split directories
    train_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_TRAIN_DIR)
    test_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_TEST_DIR)
    validation_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_VALIDATION_DIR)
    
    # Split ratios
    train_size: float = DATA_INGESTION_TRAIN_SIZE
    test_size: float = DATA_INGESTION_TEST_SIZE
    validation_size: float = DATA_INGESTION_VALIDATION_SIZE


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: str = os.path.join(ARTIFACT_DIR, PREPARE_BASE_MODEL_DIR_NAME)
    
    # Base model paths
    base_model_dir: str = os.path.join(ARTIFACT_DIR, PREPARE_BASE_MODEL_DIR_NAME, BASE_MODEL_DIR)
    base_model_file_path: str = os.path.join(ARTIFACT_DIR, PREPARE_BASE_MODEL_DIR_NAME, BASE_MODEL_DIR, BASE_MODEL_FILE_NAME)
    
    # Updated model paths
    updated_model_dir: str = os.path.join(ARTIFACT_DIR, PREPARE_BASE_MODEL_DIR_NAME, UPDATED_MODEL_DIR)
    updated_model_file_path: str = os.path.join(ARTIFACT_DIR, PREPARE_BASE_MODEL_DIR_NAME, UPDATED_MODEL_DIR, UPDATED_MODEL_FILE_NAME)
    
    # Model summary path
    model_summary_path: str = os.path.join(ARTIFACT_DIR, PREPARE_BASE_MODEL_DIR_NAME, UPDATED_MODEL_DIR, MODEL_SUMMARY_FILE_NAME)