#!/usr/bin/env python3
# app/visualize_predictions.py

from __future__ import annotations
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import yaml
from matplotlib import font_manager as fm
from typing import Tuple, List, Dict, Any


class PanoramaProcessor:
    """
    Класс для обработки панорамных изображений: нарезка на тайлы,
    выполнение предсказаний, отрисовка результатов и генерация метаданных.
    """

    def __init__(self):
        # Карта известных размеров панорам и числа тайлов
        self.SIZE_MAP = {
            (31920, 1152): 28,
            (30780, 1152): 27,
            (18144, 1142): 16,
        }
        self.FONT_SIZE = 14
        self.FONT = self._init_font()

        base_dir = Path(__file__).resolve().parent
        self.DEFAULT_WEIGHTS = str(base_dir / "weights" / "best.pt")
        self.DEFAULT_YAML = str(base_dir / "data.yaml")
        # self.DEFAULT_WEIGHTS = "weights/best.pt"
        # self.DEFAULT_YAML = "data.yaml"
        self.DEFAULT_CONF = 0.1
        # self.OUTPUT_DIR = "static/results"
        self.OUTPUT_DIR = str(base_dir / "static" / "results")

    def _init_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """
        Инициализация шрифта для отрисовки подписей.
        Пытаемся загрузить DejaVu Sans, иначе используем шрифт по умолчанию.
        """
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
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Основной метод для обработки изображения.

        1. Загружает модель YOLO.
        2. Делит панораму на тайлы.
        3. Для каждого тайла выполняет предсказание, рисует коробки и собирает метаданные.
        4. Склеивает аннотированные тайлы обратно в одну панораму.
        5. Сохраняет результат в OUTPUT_DIR.

        Возвращает:
            - путь к сохранённому файлу (str)
            - список метаданных по дефектам (list of dict)
        """
        weights = weights or self.DEFAULT_WEIGHTS
        yaml_path = yaml_path or self.DEFAULT_YAML
        conf_threshold = conf_threshold or self.DEFAULT_CONF

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Загружаем названия классов
        names = self._load_class_names(Path(yaml_path))

        # Инициализируем модель
        model = YOLO(str(weights))

        # Читаем изображение
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")

        # Делим на тайлы
        tiles = self._slice_panorama(img)

        annotated_tiles: List[np.ndarray] = []
        metadata: List[Dict[str, Any]] = []

        # Обрабатываем каждый тайл
        for idx, tile in enumerate(tiles, start=1):
            # Выполняем предсказание
            result = model.predict(tile, conf=conf_threshold, verbose=False)[0]

            # Рисуем на тайле
            annotated = self._draw_preds(tile, result, names, conf_threshold)
            annotated_tiles.append(annotated)

            # Собираем метаданные
            dets: List[Dict[str, Any]] = []
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                dets.append({
                    "class": names.get(int(box.cls[0]), str(int(box.cls[0]))),
                    "confidence": f"{conf*100:.2f}%",
                    "index": idx,
                    "coordinates": f"{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}",
                    "length": int(abs(x2 - x1))  # пиксельная длина
                })

            status = "success" if dets else "no_defects"
            metadata.append({"status": status, "defects": dets})

        # Склейка всех тайлов обратно в панораму
        result_img = self._join_tiles(annotated_tiles)

        # Сохранение
        output_path = os.path.join(self.OUTPUT_DIR, f"processed_{image_path.name}")
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

        return output_path, metadata

    def _load_class_names(self, yaml_path: Path) -> dict[int, str]:
        """
        Загрузка названий классов из YAML-файла.
        """
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in data["names"].items()}

    def _slice_panorama(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Делит панораму на список тайлов по горизонтали,
        в зависимости от её размера.
        """
        h, w = img.shape[:2]
        tiles = self.SIZE_MAP.get((w, h))
        if tiles is None:
            raise ValueError(f"Неизвестный размер панорамы {w}×{h}")
        tw = w // tiles
        return [img[:, i*tw:(i+1)*tw] for i in range(tiles)]

    def _join_tiles(self, tiles: List[np.ndarray]) -> np.ndarray:
        """
        Склеивает список тайлов обратно в одно изображение-панораму.
        """
        return np.concatenate(tiles, axis=1)

    def _draw_preds(
        self,
        tile_bgr: np.ndarray,
        res,
        names: dict[int, str],
        conf_th: float
    ) -> np.ndarray:
        """
        Рисует боксы и маски на одном тайле и возвращает результат как numpy-массив RGB.
        """
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        drw = ImageDraw.Draw(pil_img, "RGBA")

        have_masks = getattr(res, "masks", None) is not None and len(res.masks.xy) > 0

        if have_masks:
            # Если модель выдала маски
            for box, poly in zip(res.boxes, res.masks.xy):
                conf = float(box.conf[0])
                if conf < conf_th:
                    continue
                self._draw_detection(drw, poly[0], box.cls[0], names, is_mask=True)
        else:
            # Рисуем обычные боксы
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf < conf_th:
                    continue
                self._draw_detection(drw, box.xyxy[0], box.cls[0], names, is_mask=False)

        return np.asarray(pil_img)

    def _draw_detection(
        self,
        drw: ImageDraw.ImageDraw,
        coords,
        cls_id,
        names: dict[int, str],
        is_mask: bool = False
    ) -> None:
        """
        Отрисовка одного детекта: либо контур маски, либо прямоугольник.
        С подписанием класса.
        """
        cls_id = int(cls_id)
        label = names.get(cls_id, str(cls_id))

        if is_mask:
            pts = [(float(x), float(y)) for x, y in coords]
            drw.line(pts + [pts[0]], fill=(0, 255, 0, 255), width=2)
            x0, y0 = pts[0]
        else:
            x1, y1, x2, y2 = [float(v) for v in coords]
            drw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=2)
            x0, y0 = x1, y1

        tw, th = self._get_text_size(drw, label)
        # Фон под текст
        drw.rectangle([x0, y0 - th - 2, x0 + tw + 4, y0], fill=(0, 0, 0, 90))
        drw.text((x0 + 2, y0 - th - 1), label, font=self.FONT, fill=(255, 255, 255, 255))

    def _get_text_size(self, drw: ImageDraw.ImageDraw, txt: str) -> tuple[int, int]:
        """
        Получает размер текста (ширину, высоту) для корректного позиционирования.
        """
        if hasattr(drw, "textbbox"):
            x0, y0, x1, y1 = drw.textbbox((0, 0), txt, font=self.FONT)
            return x1 - x0, y1 - y0
        return self.FONT.getsize(txt)