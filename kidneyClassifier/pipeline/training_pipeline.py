import os 
import sys 
from kidneyClassifier.logger import logging
from kidneyClassifier.exception import KidneyException

from kidneyClassifier.entity.config_entity import (
    DataIngestionConfig , PrepareBaseModelConfig
)

from kidneyClassifier.entity.artifact_entity import (
    DataIngestionArtifact , PrepareBaseModelArtifact
)

from kidneyClassifier.components.data_ingestion import DataIngestion
from kidneyClassifier.components.prepare_base_model import PrepareBaseModel

from kidneyClassifier.constants import CONFIG_FILE_PATH
from kidneyClassifier.utils.common import read_yaml_file
config_contents = read_yaml_file(CONFIG_FILE_PATH)

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.prepare_base_model_config = PrepareBaseModelConfig()
        
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
    
    def start_prepare_base_model(self) -> PrepareBaseModelArtifact:
        try:
            logging.info(">>>>>>>>>>>>>Starting preparing base model<<<<<<<<<<<<<<<<<<<<")
            prepare_base_model_obj = PrepareBaseModel(
               config = config_contents,
               prepare_base_model_config = self.prepare_base_model_config
            )
            prepareBaseModelArtifact = prepare_base_model_obj.initiate_base_model()
            logging.info(
                ">>>>>>>>>>>>> preparing base model Done <<<<<<<<<<<<<<<<<<<<"
            )
            return prepareBaseModelArtifact
        except Exception as e:
            raise KidneyException(e , sys)
    
    def run_training_pipeline(self):
        """ 
        This method of TrainingPipeline class is responsible for running complete training pipeline
        """
        try:
           data_ingestion_artifact = self.start_data_ingestion()
           prepare_base_model_artifact = self.start_prepare_base_model()
           
        except Exception as e:
            raise KidneyException(e , sys)