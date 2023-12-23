import os
import joblib
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

from SolarPanelDetection import logger
from SolarPanelDetection.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, df: pd.DataFrame, config: TrainingConfig):
        self.df = df
        self.config = config

    def create_cv_splits(self, df: pd.DataFrame):
        sgkf = StratifiedGroupKFold(n_splits=self.config.n_splits, random_state=42, shuffle=True)

        self.df['fold'] = -1
        for fold, (t_idx, v_idx) in enumerate(sgkf.split(self.df, self.df['mask'], groups=self.df['image_no'])):
            self.df.loc[v_idx, 'fold'] = fold


    def one_fold(self, fold: int):
        x_train = self.df[self.df['fold']!=fold].reset_index(drop=True)
        y_train = x_train['mask'].values
        x_train = x_train[self.config.features]
        
        params = self.config.lgb_params
        model = lgb.LGBMClassifier(**params)
        model.fit(x_train, y_train)
        
        return model
    
    def save_model(self, fold: int, model: lgb.Booster):
        model_path = os.path.join(self.config.weights_dir, f"fold_{fold}.pkl")
        joblib.dump(model, model_path)

    def train(self):
        self.create_cv_splits(self.df)
        for fold in range(self.config.n_splits):
            model = self.one_fold(fold)
            self.save_model(fold, model)
