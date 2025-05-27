import numpy as np
from ultralytics import YOLO
import cv2
import os
from PIL import Image
from pathlib import Path


# -----------------------------------------------------------------
# 1) Определяем базовый каталог (папка, где находится try_model.py)
BASE_DIR = Path(__file__).resolve().parent

# 2) Формируем путь к весам относительно BASE_DIR
WEIGHTS = BASE_DIR / "app" / "weights" / "best.pt"

# 3) Грузим модель
model = YOLO(str(WEIGHTS))
# ------------------------------------------------------------------

# Создаём выходную папку
( BASE_DIR / "results" ).mkdir(exist_ok=True)

# Имя тестовой картинки
name = "t3.jpg"
path = BASE_DIR / "images" / name

results = model.predict(str(path))

SIZE_MAP = {
    (31920, 1152): 28,
    (30780, 1152): 27,
    (18144, 1142): 16,
}

image = Image.open(path)
size = image.size

# print(results[0].orig_img)
# Вывод предсказаний
print("\nРезультаты обнаружения:")
index = 1
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = result.names[class_id]
        confidence = float(box.conf)
        bbox = [round(x) for x in box.xyxy[0].tolist()]

        print(f"Обнаружен: {class_name}")
        print(f"Уверенность: {confidence:.2%}")
        print(f"Координаты: x1={bbox[0]}, y1={bbox[3]}, y2={bbox[1]}, x2={bbox[2]}")
        print("-" * 30)

# Обработка и сохранение результатов
for result in results:
    annotated_img = result.plot()  # Получаем изображение с разметкой

    # Конвертируем RGB в BGR для OpenCV
    annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

    # Показываем изображение
    cv2.imshow("Detection Results", annotated_img_bgr)
    cv2.waitKey(0)

    # Сохраняем результат с полным путем
    output_path = os.path.join("results", f"processed_{name}")
    cv2.imwrite(output_path, annotated_img_bgr)
    print(f"Изображение сохранено в: {output_path}")