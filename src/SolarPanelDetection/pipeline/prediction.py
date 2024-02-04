import os
import joblib
import numpy as np
from PIL import Image
from SolarPanelDetection.utils.common import read_tiff

class PredictionPipeline:
    def __init__(self):
        self.folds=5

    def load_model(self, fold: int):
        return joblib.load(f"models/fold_{fold}.pkl")
    
    def predict(self, filename):
        img = read_tiff(filename)
        reshaped_img = img.reshape(-1, 12)
        oof_test_preds = []
        for fold in range(self.folds):
            model = self.load_model(fold)
            probs = model.predict_proba(reshaped_img)
            oof_test_preds.append(probs)

        mask = np.mean(oof_test_preds, axis=0)
        mask = np.argmax(mask, axis=1)

        mask = mask.reshape(img.shape[0], img.shape[1])

        return mask
    
