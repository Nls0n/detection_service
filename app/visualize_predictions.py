#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import yaml
from matplotlib import font_manager as fm


class PanoramaProcessor:
    def __init__(self):
        # Конфигурация по умолчанию
        self.SIZE_MAP = {
            (31920, 1152): 28,
            (30780, 1152): 27,
            (18144, 1142): 16,
        }
        self.FONT_SIZE = 14
        self.FONT = self._init_font()

        # Пути по умолчанию (можно переопределить)
        self.DEFAULT_WEIGHTS = "weights/model.pt"
        self.DEFAULT_YAML = "data.yaml"
        self.DEFAULT_CONF = 0.15
        self.OUTPUT_DIR = "static/results"

    def _init_font(self):
        """Инициализация шрифта"""
        try:
            path = fm.findfont("DejaVu Sans")
            return ImageFont.truetype(path, self.FONT_SIZE)
        except Exception:
            return ImageFont.load_default()

    def process_image(
            self,
            image_path: str | Path,
            weights: str | Path = None,
            yaml_path: str | Path = None,
            conf_threshold: float = None
    ) -> str:
        """
        Основной метод для обработки изображения
        Возвращает путь к обработанному изображению
        """
        # Установка значений по умолчанию
        weights = weights or self.DEFAULT_WEIGHTS
        yaml_path = yaml_path or self.DEFAULT_YAML
        conf_threshold = conf_threshold or self.DEFAULT_CONF

        # Создаем выходную директорию
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Загрузка модели и классов
        names = self._load_class_names(yaml_path)
        model = YOLO(str(weights))

        # Обработка изображения
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")

        tiles = self._slice_panorama(img)
        processed_tiles = []

        for tile in tiles:
            res = model.predict(tile, conf=conf_threshold, verbose=False)[0]
            processed_tiles.append(self._draw_preds(tile, res, names, conf_threshold))

        # Склейка и сохранение результата
        result_img = self._join_tiles(processed_tiles)
        output_path = os.path.join(self.OUTPUT_DIR, f"processed_{image_path.name}")
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

        return output_path

    def _load_class_names(self, yaml_path: Path) -> dict[int, str]:
        """Загрузка названий классов из YAML"""
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        return {int(k): v for k, v in data["names"].items()}

    def _slice_panorama(self, img: np.ndarray) -> list[np.ndarray]:
        """Нарезка панорамы на тайлы"""
        h, w = img.shape[:2]
        tiles = self.SIZE_MAP.get((w, h))
        if tiles is None:
            raise ValueError(f"Неизвестный размер панорамы {w}×{h}")
        tw = w // tiles
        return [img[:, i * tw:(i + 1) * tw] for i in range(tiles)]

    def _join_tiles(self, tiles: list[np.ndarray]) -> np.ndarray:
        """Склейка тайлов обратно в панораму"""
        return np.concatenate(tiles, axis=1)

    def _draw_preds(
            self,
            tile_bgr: np.ndarray,
            res,
            names: dict[int, str],
            conf_th: float
    ) -> np.ndarray:
        """Отрисовка предсказаний на одном тайле"""
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        drw = ImageDraw.Draw(pil, "RGBA")

        have_masks = res.masks is not None and len(res.masks.xy) > 0

        if have_masks:
            for box, poly in zip(res.boxes, res.masks.xy):
                conf = float(box.conf[0])
                if conf < conf_th:
                    continue
                self._draw_detection(drw, poly[0], box.cls[0], names, is_mask=True)
        else:
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf < conf_th:
                    continue
                self._draw_detection(drw, box.xyxy[0], box.cls[0], names)

        return np.asarray(pil)

    def _draw_detection(self, drw, coords, cls_id, names, is_mask=False):
        """Отрисовка одного обнаружения"""
        cls_id = int(cls_id)
        label = names.get(cls_id, str(cls_id))

        if is_mask:
            # Для масок (контуры)
            pts = [(float(x), float(y)) for x, y in coords]
            drw.line(pts + [pts[0]], fill=(0, 255, 0, 255), width=2)
            x0, y0 = pts[0]
        else:
            # Для bounding box
            x1, y1, x2, y2 = [float(v) for v in coords]
            drw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=2)
            x0, y0 = x1, y1

        # Отрисовка подписи
        tw, th = self._get_text_size(drw, label)
        drw.rectangle([x0, y0 - th - 2, x0 + tw + 4, y0], fill=(0, 0, 0, 90))
        drw.text((x0 + 2, y0 - th - 1), label, font=self.FONT, fill=(255, 255, 255, 255))

    def _get_text_size(self, drw, txt: str) -> tuple[int, int]:
        """Получение размера текста"""
        if hasattr(drw, "textbbox"):
            x0, y0, x1, y1 = drw.textbbox((0, 0), txt, font=self.FONT)
            return x1 - x0, y1 - y0
        return self.FONT.getsize(txt)


# Пример использования:
if __name__ == "__main__":
    processor = PanoramaProcessor()

    # Самый простой вариант вызова - только путь к изображению
    result_path = processor.process_image("temp_uploads/926db102-5268-4120-bcbe-d5b2d1472244.jpg")
    print(f"Результат сохранён: {result_path}")

    # Вариант с переопределением параметров
    # result_path = processor.process_image(
    #     image_path="path/to/image.png",
    #     weights="custom_weights.pt",
    #     yaml_path="custom_data.yaml",
    #     conf_threshold=0.25
    # )