import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import f1_score, log_loss

import mlflow
from urllib.parse import urlparse
from SolarPanelDetection import logger
from SolarPanelDetection.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self,df: pd.DataFrame, config: EvaluationConfig):
        self.config = config
        self.df = df

    def one_fold(self,fold: int, df: pd.DataFrame):
        x_val = self.df[self.df['fold']!=fold].reset_index(drop=True)
        y_val = x_val['mask'].values
        x_val = x_val[self.config.all_params.features]

        self.load_model(fold)
        val_prob = self.model.predict_proba(x_val)
        val_preds = np.argmax(val_prob, axis=1)
        val_f1 = f1_score(y_val, val_preds)
        val_lloss = log_loss(y_val, val_preds)

        return {
            'f1_score': val_f1,
            'log_loss': val_lloss
        }
    
    def load_model(self, fold: int):
        model_path = os.path.join(self.config.weights_dir, f"fold_{fold}.pkl")
        self.model = joblib.load(model_path)
    
    def evaluate(self):
        for fold in range(self.config.all_params.n_splits):
            self.scores = self.one_fold(fold, self.df)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"f1_score": self.scores['f1_score'], "log_loss": self.scores['log_loss']}
            )
            
            if tracking_url_type_store != "file":
                mlflow.lightgbm.log_model(self.model, "model", registered_model_name="Lightgbm")
            else:
                mlflow.lightgbm.log_model(self.model, "model")

