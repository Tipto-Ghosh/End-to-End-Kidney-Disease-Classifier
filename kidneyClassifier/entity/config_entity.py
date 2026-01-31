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