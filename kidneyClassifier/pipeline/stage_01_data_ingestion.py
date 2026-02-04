import os 
import sys 
from pathlib import Path
from kidneyClassifier.logger import logging
from kidneyClassifier.exception import KidneyException
from kidneyClassifier.entity.config_entity import DataIngestionConfig
from kidneyClassifier.entity.artifact_entity import DataIngestionArtifact
from kidneyClassifier.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion Stage"
ARTIFACT_FILE = "artifacts/data_ingestion/artifact.json"

class DataIngestionTrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def main(self) -> DataIngestionArtifact:
        """
        Execute the data ingestion pipeline stage
        """
        try:
            logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            
            data_ingestion_obj = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion_obj.initiate_data_ingestion()
            
            # Save artifact metadata
            data_ingestion_artifact.save(ARTIFACT_FILE)
            logging.info(f"Artifact metadata saved to: {ARTIFACT_FILE}")
            
            logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact
            
        except Exception as e:
            logging.exception(e)
            raise KidneyException(e, sys)



if __name__ == '__main__':
    try:
        obj = DataIngestionTrainingPipeline()
        artifact = obj.main()
    except Exception as e:
        logging.exception(e)
        raise e