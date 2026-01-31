from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Data Ingestion Artifact containing paths to ingested data
    """
    extracted_data_path: str
    downloaded_file_path: str