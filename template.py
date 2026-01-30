import os
from pathlib import Path
import logging


logging.basicConfig(level = logging.INFO, format = '[%(asctime)s]: %(message)s')

project_name = 'kidneyClassifier'

files_list = [
    ".github/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "requirements.txt",
    "setup.py",
    "notebooks/test.ipynb",
    "templates/index.html"
]


for filepath in files_list:
    filepath = Path(filepath)
    file_dir , file_name = os.path.split(filepath)
    
    if file_dir != "":
        os.makedirs(file_dir , exist_ok = True)
        logging.info(f"creating dicrectory: {file_dir} for the file: {file_name}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath , "w") as f:
            pass 
            logging.info(f"creating empty file: {file_name}")
    else:
        logging.info(f"{file_name} is already exists")