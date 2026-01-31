import os 
import sys 
from kidneyClassifier.logger import logging
from kidneyClassifier.exception import KidneyException
from kidneyClassifier.entity.artifact_entity import DataIngestionArtifact
from kidneyClassifier.entity.config_entity import DataIngestionConfig
from kidneyClassifier.components.data_ingestion import DataIngestion

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(">>>>>>>>>>>>>Starting data ingestion<<<<<<<<<<<<<<<<<<<<")
            data_ingestion_obj = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion_obj.initiate_data_ingestion()
            logging.info(
                ">>>>>>>>>>>>> Data Ingestion Done <<<<<<<<<<<<<<<<<<<<"
            )
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise KidneyException(e , sys)
    
    def run_training_pipeline(self):
        """ 
        This method of TrainingPipeline class is responsible for running complete training pipeline
        """
        try:
           data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise KidneyException(e , sys)