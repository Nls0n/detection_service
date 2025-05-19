from ultralytics import YOLO
from typing import List, Dict, Any
import numpy as np


class DefectDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.classes = self.model.names

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Обработка изображения с конвертацией numpy в python-типы"""
        results = self.model(image, verbose=False)
        detections = []

        for box in results[0].boxes:
            detections.append({
                "class": self.classes[int(box.cls)],
                "confidence": float(box.conf),  # Явное преобразование в float
            })

        return {
            "status": "success" if detections else "no_defects",
            "detections": detections,
        }