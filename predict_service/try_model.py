from ultralytics import YOLO
import cv2
import os

# Создаем папку results, если её нет
os.makedirs("results", exist_ok=True)

# Инициализация модели
model = YOLO(r"C:\codes\AI_mission\app\weights\last.pt")

# Предсказание на тестовом изображении
name = f'10-310-ls-34-g-01.png'
path = f"images/{name}"
results = model.predict(path)

# print(results[0].orig_img)
# Вывод предсказаний
print("\nРезультаты обнаружения:")
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = result.names[class_id]
        confidence = float(box.conf)
        bbox = [round(x) for x in box.xyxy[0].tolist()]

        print(f"Обнаружен: {class_name}")
        print(f"Уверенность: {confidence:.2%}")
        print(f"Координаты: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")
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