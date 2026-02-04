from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Data Ingestion Artifact containing paths to ingested and split data
    """
    extracted_data_path: str
    downloaded_file_path: str
    train_data_path: str
    test_data_path: str
    validation_data_path: str
    is_artifact_valid: bool = True 
    
    def save(self, filepath: Path):
        """Save artifact metadata to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        artifact_dict = {
            'extracted_data_path': str(self.extracted_data_path),
            'downloaded_file_path': str(self.downloaded_file_path),
            'train_data_path': str(self.train_data_path),
            'test_data_path': str(self.test_data_path),
            'validation_data_path': str(self.validation_data_path),
            'is_artifact_valid': self.is_artifact_valid
        }
        
        with open(filepath, 'w') as f:
            json.dump(artifact_dict, f, indent=4)
    
    @classmethod
    def load(cls, filepath: Path):
        """Load artifact metadata from JSON file"""
        with open(filepath, 'r') as f:
            artifact_dict = json.load(f)
        
        return cls(
            extracted_data_path=artifact_dict['extracted_data_path'],
            downloaded_file_path=artifact_dict['downloaded_file_path'],
            train_data_path=artifact_dict['train_data_path'],
            test_data_path=artifact_dict['test_data_path'],
            validation_data_path=artifact_dict['validation_data_path'],
            is_artifact_valid=artifact_dict['is_artifact_valid']
        )
        
@dataclass(frozen=True)
class PrepareBaseModelArtifact:
    """
    Prepare Base Model Artifact containing paths to base and updated models
    """
    base_model_path: str
    updated_model_path: str
    model_summary_path: str
    is_artifact_valid: bool = True
    
    def save(self, filepath: Path):
        """Save artifact metadata to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        artifact_dict = {
            'base_model_path': str(self.base_model_path),
            'updated_model_path': str(self.updated_model_path),
            'model_summary_path': str(self.model_summary_path),
            'is_artifact_valid': self.is_artifact_valid
        }
        
        with open(filepath, 'w') as f:
            json.dump(artifact_dict, f, indent=4)
    
    @classmethod
    def load(cls, filepath: Path):
        """Load artifact metadata from JSON file"""
        with open(filepath, 'r') as f:
            artifact_dict = json.load(f)
        
        return cls(
            base_model_path=artifact_dict['base_model_path'],
            updated_model_path=artifact_dict['updated_model_path'],
            model_summary_path=artifact_dict['model_summary_path'],
            is_artifact_valid=artifact_dict['is_artifact_valid']
        )