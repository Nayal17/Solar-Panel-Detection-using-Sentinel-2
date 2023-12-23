import os
import gdown
import zipfile
from pathlib import Path
from SolarPanelDetection import logger
from SolarPanelDetection.config.configuration import ConfigurationManager
from SolarPanelDetection.utils.common import get_size, create_directories
from SolarPanelDetection.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        '''
        Fetch data from the URL
        '''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            file_id = dataset_url.split("/")[-2]
            prefix =  'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        create_directories([Path(unzip_path)])
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

