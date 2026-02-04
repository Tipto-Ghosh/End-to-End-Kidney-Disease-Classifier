import os 
import sys 
from pathlib import Path
from kidneyClassifier.logger import logging
from kidneyClassifier.exception import KidneyException
from kidneyClassifier.entity.config_entity import PrepareBaseModelConfig
from kidneyClassifier.entity.artifact_entity import PrepareBaseModelArtifact
from kidneyClassifier.components.prepare_base_model import PrepareBaseModel
from kidneyClassifier.constants import CONFIG_FILE_PATH
from kidneyClassifier.utils.common import read_yaml_file


STAGE_NAME = "Prepare Base Model Stage"
ARTIFACT_FILE = "artifacts/prepare_base_model/artifact.json"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        self.prepare_base_model_config = PrepareBaseModelConfig()
        self.config_contents = read_yaml_file(CONFIG_FILE_PATH)
        
    def main(self) -> PrepareBaseModelArtifact:
        """
        Execute the prepare base model pipeline stage
        """
        try:
            logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            
            prepare_base_model_obj = PrepareBaseModel(
                config = self.config_contents,
                prepare_base_model_config = self.prepare_base_model_config
            )
            prepare_base_model_artifact = prepare_base_model_obj.initiate_base_model()
            
            # Save artifact metadata
            prepare_base_model_artifact.save(ARTIFACT_FILE)
            logging.info(f"Artifact metadata saved to: {ARTIFACT_FILE}")
            
            logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
            logging.info(f"Prepare Base Model artifact: {prepare_base_model_artifact}")
            
            return prepare_base_model_artifact
            
        except Exception as e:
            logging.exception(e)
            raise KidneyException(e, sys)


if __name__ == '__main__':
    try:
        obj = PrepareBaseModelTrainingPipeline()
        artifact = obj.main()
    except Exception as e:
        logging.exception(e)
        raise e