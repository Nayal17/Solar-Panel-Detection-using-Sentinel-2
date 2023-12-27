import os
import yaml
import json
import shutil
import tifffile
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from SolarPanelDetection import logger
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations


@ensure_annotations
def read_yaml(path_to_yaml: Path)-> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): logs directory creation
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def copy_tree(src: str, dst: str, verbose=True):
    """copy a directory

    Args:
        src: source directory
        dst: destination directory
        verbose: logs src and dst paths
    """
    if not os.path.exists(dst):
        shutil.copytree(src, dst)

    if verbose:
        logger.info(f"Copied files from {src} to {dst}")


@ensure_annotations
def read_tiff(path: Path) -> np.ndarray:
    """reads tiff file and returns

    Args:
        path (str): path like input

    Raises:
        e: if tiff file doesn't exist

    Returns:
        img: np.ndarray
    """
    try:
        img = tifffile.imread(path)
        logger.info(f"tiff file: {path} loaded successfully")
        return img

    except Exception as e:
        raise e
    
@ensure_annotations
def read_csv(path: Path) -> pd.DataFrame:
    """reads csv file and returns

    Args:
        path (str): path like input

    Raises:
        e: if csv file doesn't exist

    Returns:
        img: pd.DataFrame
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"csv file: {path} loaded successfully")
        return df

    except Exception as e:
        raise e

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"
