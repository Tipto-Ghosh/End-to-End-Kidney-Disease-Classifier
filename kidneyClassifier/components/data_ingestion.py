import sys
import os
import zipfile
import shutil
from pathlib import Path
from typing import List, Tuple
import random
from kidneyClassifier.logger import logging
from kidneyClassifier.exception import KidneyException
import gdown
from kidneyClassifier.entity.artifact_entity import DataIngestionArtifact
from kidneyClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(
        self, 
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        self.data_ingestion_config = data_ingestion_config
    
    def download_dataset(self):
        """
        Download dataset from google drive
        """
        try:
            logging.info("Entering download_dataset method inside DataIngestion")
            
            dataset_url = self.data_ingestion_config.source_url
            zip_download_dir = self.data_ingestion_config.local_data_file
            
            os.makedirs(self.data_ingestion_config.root_dir, exist_ok=True)
            
            logging.info(f"Starting downloading the dataset from: {dataset_url}")
            gdown.download(
                url=dataset_url,
                output=zip_download_dir,
                fuzzy=True  
            )
            logging.info(f"Dataset downloaded to: {zip_download_dir}")
            
        except Exception as e:
            logging.error(f"Failed to download dataset from google drive: {str(e)}")
            raise KidneyException(e, sys)
    
    def extract_zip_file(self):
        """
        Extract zip file to unzip directory
        """
        try:
            logging.info("Entering extract_zip_file method")
            
            unzip_path = self.data_ingestion_config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            
            logging.info(f"Extracting {self.data_ingestion_config.local_data_file} to {unzip_path}")
            
            with zipfile.ZipFile(self.data_ingestion_config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            
            logging.info("Extraction completed successfully")
            
        except Exception as e:
            logging.error(f"Failed to extract zip file: {str(e)}")
            raise KidneyException(e, sys)
    
    def get_image_files(self, class_dir: Path) -> List[Path]:
        """
        Get all image files from a directory
        
        Args:
            class_dir: Directory containing images
            
        Returns:
            List of image file paths
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        return image_files
    
    def split_data(self):
        """
        Split data into train, test, and validation sets
        
        Expected structure:
        extracted_data/kidney-ct-scan-image/
            ├── normal/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            └── Tumor/
                ├── image1.jpg
                ├── image2.jpg
                └── ...
        
        Output structure:
        train/
            ├── normal/
            └── Tumor/
        test/
            ├── normal/
            └── Tumor/
        validation/
            ├── normal/
            └── Tumor/
        """
        try:
            logging.info("Starting data splitting process")
            
            # Find the kidney-ct-scan-image directory
            extracted_dir = Path(self.data_ingestion_config.unzip_dir)
            kidney_dir = extracted_dir / "kidney-ct-scan-image"
            
            if not kidney_dir.exists():
                raise FileNotFoundError(f"Directory not found: {kidney_dir}")
            
            # Get class directories (normal and Tumor)
            class_dirs = [d for d in kidney_dir.iterdir() if d.is_dir()]
            
            if len(class_dirs) == 0:
                raise ValueError(f"No class directories found in {kidney_dir}")
            
            logging.info(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
            
            # Create split directories
            train_dir = Path(self.data_ingestion_config.train_dir)
            test_dir = Path(self.data_ingestion_config.test_dir)
            val_dir = Path(self.data_ingestion_config.validation_dir)
            
            # Get split ratios
            train_ratio = self.data_ingestion_config.train_size
            test_ratio = self.data_ingestion_config.test_size
            val_ratio = self.data_ingestion_config.validation_size
            
            # Validate ratios
            assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-5, \
                "Train, test, and validation ratios must sum to 1.0"
            
            # Process each class
            for class_dir in class_dirs:
                class_name = class_dir.name
                logging.info(f"Processing class: {class_name}")
                
                # Create class directories in train/test/val
                (train_dir / class_name).mkdir(parents=True, exist_ok=True)
                (test_dir / class_name).mkdir(parents=True, exist_ok=True)
                (val_dir / class_name).mkdir(parents=True, exist_ok=True)
                
                # Get all image files
                image_files = self.get_image_files(class_dir)
                total_images = len(image_files)
                
                logging.info(f"Found {total_images} images in {class_name}")
                
                if total_images == 0:
                    logging.warning(f"No images found in {class_dir}")
                    continue
                
                # Shuffle images for random split
                random.seed(42)  # For reproducibility
                random.shuffle(image_files)
                
                # Calculate split indices
                train_end = int(total_images * train_ratio)
                test_end = train_end + int(total_images * test_ratio)
                
                # Split files
                train_files = image_files[:train_end]
                test_files = image_files[train_end:test_end]
                val_files = image_files[test_end:]
                
                logging.info(f"{class_name} split: Train={len(train_files)}, "
                           f"Test={len(test_files)}, Validation={len(val_files)}")
                
                # Copy files to respective directories
                for img_file in train_files:
                    shutil.copy2(img_file, train_dir / class_name / img_file.name)
                
                for img_file in test_files:
                    shutil.copy2(img_file, test_dir / class_name / img_file.name)
                
                for img_file in val_files:
                    shutil.copy2(img_file, val_dir / class_name / img_file.name)
            
            logging.info("Data splitting completed successfully")
            
            # Log summary
            self._log_split_summary()
            
        except Exception as e:
            logging.error(f"Failed to split data: {str(e)}")
            raise KidneyException(e, sys)
    
    def _log_split_summary(self):
        """Log summary of data split"""
        try:
            train_dir = Path(self.data_ingestion_config.train_dir)
            test_dir = Path(self.data_ingestion_config.test_dir)
            val_dir = Path(self.data_ingestion_config.validation_dir)
            
            logging.info("=" * 60)
            logging.info("DATA SPLIT SUMMARY")
            logging.info("=" * 60)
            
            for split_name, split_dir in [("Train", train_dir), ("Test", test_dir), ("Validation", val_dir)]:
                if split_dir.exists():
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            count = len(list(class_dir.iterdir()))
                            logging.info(f"{split_name}/{class_dir.name}: {count} images")
            
            logging.info("=" * 60)
            
        except Exception as e:
            logging.warning(f"Could not log split summary: {str(e)}")
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates data ingestion process
        
        Returns:
            DataIngestionArtifact: Contains paths to downloaded, extracted, and split data
        """
        try:
            logging.info("=" * 60)
            logging.info("STARTING DATA INGESTION")
            logging.info("=" * 60)
            
            # Step 1: Download dataset
            logging.info("Step 1/3: Downloading dataset...")
            self.download_dataset()
            
            # Step 2: Extract zip file
            logging.info("Step 2/3: Extracting dataset...")
            self.extract_zip_file()
            
            # Step 3: Split data into train/test/validation
            logging.info("Step 3/3: Splitting data into train/test/validation...")
            self.split_data()
            
            # Create artifact
            data_ingestion_artifact = DataIngestionArtifact(
                extracted_data_path=self.data_ingestion_config.unzip_dir,
                downloaded_file_path=self.data_ingestion_config.local_data_file,
                train_data_path=self.data_ingestion_config.train_dir,
                test_data_path=self.data_ingestion_config.test_dir,
                validation_data_path=self.data_ingestion_config.validation_dir,
                is_artifact_valid=True
            )
            
            logging.info("=" * 60)
            logging.info("DATA INGESTION COMPLETED SUCCESSFULLY")
            logging.info("=" * 60)
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise KidneyException(e, sys)