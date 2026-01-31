import os
import yaml
from kidneyClassifier.logger import logging
from kidneyClassifier.exception import KidneyException
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
import base64
import sys
import json




@ensure_annotations
def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its content as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed content of the YAML file.

    Raises:
        LaptopException: If reading the YAML file fails.
    """
    
    logging.info(f"Entered read_yaml_file with file_path={file_path}")
    try:
        with open(file_path, "rb") as yaml_file:
            data = yaml.safe_load(yaml_file)
        logging.info(f"YAML file loaded successfully: {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error occurred while reading YAML file: {file_path}")
        raise KidneyException(e, sys)  
    

@ensure_annotations
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write content to a YAML file.

    Args:
        file_path (str): Path where the YAML file should be saved.
        content (object): The content to write (usually a dict).
        replace (bool, optional): Whether to replace an existing file. Defaults to False.

    Raises:
        LaptopException: If writing the YAML file fails.
    """
    
    logging.info(f"Entered write_yaml_file with file_path={file_path}, replace={replace}")
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Existing file removed: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        logging.info(f"YAML file written successfully: {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while writing YAML file: {file_path}")
        raise KidneyException(e, sys)  

@ensure_annotations
def save_yaml_file(file_path: str, data: dict):
    """
    Save dictionary data to a YAML file.

    Args:
        file_path (str): Path where the YAML file should be saved.
        data (dict): Dictionary data to save.

    Raises:
        LaptopException: If saving the YAML file fails.
    """
    logging.info(f"Entered save_yaml_file with file_path={file_path}")
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'w') as file_obj:
            yaml.dump(data, file_obj)
        logging.info(f"YAML file saved successfully: {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving YAML file: {file_path}")
        raise KidneyException(e, sys)  


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> dict:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logging.info(f"json file loaded succesfully from: {path}")
    return content

@ensure_annotations
def get_file_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

