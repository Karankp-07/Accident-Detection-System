import os
from typing import Tuple
import numpy as np
from tensorflow.keras.models import model_from_json


class AccidentDetectionModel:
    class_nums = ['Accident', 'No Accident']

    def __init__(self, model_json_path: str, model_weights_path: str):
        if not os.path.exists(model_json_path):
            raise FileNotFoundError(f"Model architecture file not found: {model_json_path}")
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights file not found: {model_weights_path}")

        with open(model_json_path, "r", encoding="utf-8") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_path)
        self.loaded_model.make_predict_function()

    def predict_accident(self, img_batch: np.ndarray) -> Tuple[str, np.ndarray]:
        preds = self.loaded_model.predict(img_batch, verbose=0)
        return AccidentDetectionModel.class_nums[int(np.argmax(preds))], preds