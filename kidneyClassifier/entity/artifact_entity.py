from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Data Ingestion Artifact containing paths to ingested data
    """
    extracted_data_path: str
    downloaded_file_path: str

@dataclass(frozen = True)
class PrepareBaseModelArtifact:
    base_model_file_path: str 
    updated_base_model_file_path: str
    updated_model_architecture_summary_file_path: str