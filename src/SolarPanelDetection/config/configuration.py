import os
from SolarPanelDetection import logger
from SolarPanelDetection.constants import *
from SolarPanelDetection.utils.common import read_yaml, create_directories
from SolarPanelDetection.entity.config_entity import (DataIngestionConfig,
                                                      DataPreparationConfig,
                                                      TrainingConfig,
                                                      EvaluationConfig,
                                                      PredictionConfig
                                                    )


class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])
        
    def get_data_ingestion_config(self)->DataIngestionConfig:

        create_directories([Path(self.config.data_ingestion.root_dir)])

        data_ingestion_config = DataIngestionConfig(
            root_dir=self.config.data_ingestion.root_dir,
            source_URL=self.config.data_ingestion.source_URL,
            local_data_file=self.config.data_ingestion.local_data_file,
            unzip_dir=self.config.data_ingestion.unzip_dir 
        )

        return data_ingestion_config

    def get_data_preparation_config(self) -> DataPreparationConfig:
        
        create_directories([Path(self.config.data_preparation.root_dir)])

        get_data_preparation_config = DataPreparationConfig(
            features=self.params.features,
            n_splits=self.params.n_splits,
            img_dir=os.path.join(self.config.data_ingestion.unzip_dir, "s2_image"),
            mask_dir=os.path.join(self.config.data_ingestion.unzip_dir, "mask"),
            dataframe_save_path = self.config.data_preparation.df_save_path
        )

        return get_data_preparation_config
    
    def get_training_config(self) -> TrainingConfig:

        create_directories([Path(self.config.training.root_dir), Path(self.config.training.weights_dir)])
        prepare_training_config = TrainingConfig(
            features=self.params.features,
            n_splits=self.params.n_splits,
            lgb_params=self.params.lgb_params,
            weights_dir=self.config.training.weights_dir,
            df_path = self.config.data_preparation.df_save_path
        )

        return prepare_training_config

    def get_evaluation_config(self) -> EvaluationConfig:

        eval_config = EvaluationConfig(
            weights_dir=self.config.training.weights_dir,
            df_path=self.config.data_preparation.df_save_path,
            mlflow_uri=os.environ['MLFLOW_TRACKING_URI'],
            all_params=self.params,
        )
        return eval_config
    
    def get_prediction_config(self) -> PredictionConfig:

        predict_config = PredictionConfig(
            n_splits=self.params.n_splits,
        )
        return predict_config