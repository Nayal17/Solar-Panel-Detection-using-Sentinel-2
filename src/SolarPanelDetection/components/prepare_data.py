import os
import numpy as np
import pandas as pd
from pathlib import Path
from SolarPanelDetection import logger
from SolarPanelDetection.utils.common import read_tiff 
from SolarPanelDetection.entity.config_entity import DataPreparationConfig


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def get_features(self):
        img_dir = self.config.img_dir
        mask_dir = self.config.mask_dir
        features = self.config.features
        dataframe_save_path = self.config.dataframe_save_path
        image_names = sorted(os.listdir(img_dir))
        mask_names = sorted(os.listdir(mask_dir))

        data_list=[]
        for i, (img_name, mask_name) in enumerate(zip(image_names, mask_names)):
            img = read_tiff(Path(os.path.join(img_dir, img_name))).astype(float)
            mask = read_tiff(Path(os.path.join(mask_dir, mask_name))).astype(float)

            reshaped_img = img.reshape(-1, 12) # (23, 23, 12) -> (23*23, 12)
            flatten_mask = mask.reshape(-1,1)   # (23, 23) -> (23*23, 1)

            image_no = np.full((reshaped_img.shape[0], 1), i)
            combine_data = np.hstack((reshaped_img, flatten_mask))
            combine_data = np.hstack((combine_data, image_no))
            data_list.append(combine_data)

        data_list = np.vstack(data_list)
        df = pd.DataFrame(data_list, columns=features+['mask', 'image_no'])
        df.to_csv(dataframe_save_path, index=False)
        logger.info("Features dataframe created")
