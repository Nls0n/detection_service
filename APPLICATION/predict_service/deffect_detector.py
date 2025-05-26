from ultralytics import YOLO
from typing import Dict, Any
import numpy as np


class DefectDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.classes = self.model.names
        self.index = 1

    def predict(self, image: np.ndarray, panorama_size: tuple=(31920, 1152), index: int=1) -> Dict[str, Any]:
        """Обработка изображения с конвертацией numpy в python-типы"""
        results = self.model(image, conf=0.1, verbose=False)
        size = panorama_size
        detections = []
        SIZE_MAP = {
            (31920, 1152): 28,
            (30780, 1152): 27,
            (18144, 1142): 16,
        }

        for box in results[0].boxes:
            bbox = [round(x) for x in box.xyxy[0].tolist()]
            x1 = bbox[0] + self.index * size[0] / SIZE_MAP[size]
            y1 = bbox[3]
            x2 = bbox[2] + self.index * size[0] / SIZE_MAP[size]
            y2 = bbox[1]
            length = (int((x1+x2-2000)/2*310/size[0]))%310
            if length%10>=5:
                length+=10-length%10
            else:
                length-=length%10
            detections.append({
                "class": self.classes[int(box.cls)],
                "confidence": float(box.conf),  # Явное преобразование в float
                "coordinates": f"{x1=}, {y1=}, {x2=}, {y2=}",
                "index": index,
                "length": length
            })
        self.index+=1

        return {
            "status": "success" if detections else "no_defects",
            "detections": detections,
        }