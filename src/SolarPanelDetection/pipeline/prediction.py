import os
import joblib
import numpy as np
from SolarPanelDetection.utils.common import read_tiff
from SolarPanelDetection.entity.config_entity import PredictionConfig

class PredictionPipeline:
    def __init__(self, config: PredictionConfig):
        self.config = config

    def load_model(self, fold: int):
        return joblib.load(f"models/fold_{fold}.pkl")
    
    def predict(self, filename):
        img = read_tiff(filename)
        reshaped_img = img.reshape(-1, 12)
        oof_test_preds = []
        for fold in range(self.config.n_splits):
            model = self.load_model(fold)
            probs = model.predict_proba(reshaped_img)
            oof_test_preds.append(probs)

        mask = np.mean(oof_test_preds, axis=0)
        mask = mask.reshape(img.shape[0], img.shape[1])

        return mask
