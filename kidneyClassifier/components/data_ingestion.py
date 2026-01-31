import sys
import os
import zipfile
from kidneyClassifier.logger import logging
from kidneyClassifier.exception import KidneyException
import gdown
from kidneyClassifier.entity.artifact_entity import DataIngestionArtifact
from kidneyClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(
        self , 
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        self.data_ingestion_config = data_ingestion_config
    
    def download_dataset(self):
        """
        Download dataset from the google drive
        """
        try:
            logging.info(
                "Entering download_dataset method inside DataIngestion"
            )
            dataset_url = self.data_ingestion_config.source_url
            zip_download_dir = self.data_ingestion_config.local_data_file
            os.makedirs(self.data_ingestion_config.root_dir)
            logging.info("Starting downloading the dataset from gdrive.")
            gdown.download(
                url = dataset_url,
                output = zip_download_dir,
                fuzzy = True  
            )
            logging.info("Dataset download completed")
        except Exception as e:
            logging.info(f"Failed to download dataset from the google drive")
            raise KidneyException(e , sys)
    
    def extract_zip_file(self):
        unzip_path = self.data_ingestion_config.unzip_dir
        os.makedirs(unzip_path , exist_ok = True)
        with zipfile.ZipFile(self.data_ingestion_config.local_data_file , 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates data ingestion process
        Returns:
            DataIngestionArtifact: Contains paths to downloaded and extracted data
        """
        try:
            self.download_dataset()
            self.extract_zip_file()
            
            data_ingestion_artifact = DataIngestionArtifact(
                extracted_data_path=self.data_ingestion_config.unzip_dir,
                downloaded_file_path=self.data_ingestion_config.local_data_file
            )
            
            logging.info(f"Data ingestion completed successfully")
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise KidneyException(e, sys)