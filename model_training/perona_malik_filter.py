import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil


def perona_malik_filter(image, iterations=100, delta_t=0.1, kappa=30, option=1):
    """
    Применяет анизотропную диффузию Перона-Малика к изображению.
    """
    if iterations == 0:
        return image.copy()

    # Конвертируем numpy массив в cupy массив
    img = np.asarray(image, dtype=np.float32)

    for _ in range(iterations):
        grad_x = np.roll(img, -1, axis=1) - np.roll(img, 1, axis=1)
        grad_y = np.roll(img, -1, axis=0) - np.roll(img, 1, axis=0)
        grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)

        if option == 1:
            c = 1.0 / (1.0 + (grad_norm / kappa) ** 2)
        elif option == 2:
            c = np.exp(-(grad_norm / kappa) ** 2)
        else:
            raise ValueError("option must be 1 or 2")

        img += delta_t * (
                c * (np.roll(img, -1, axis=1) + np.roll(img, 1, axis=1) - 2 * img) +
                c * (np.roll(img, -1, axis=0) + np.roll(img, 1, axis=0) - 2 * img)
        )

    # Конвертируем обратно в numpy массив перед возвратом
    return img


def process_images(input_folder='dataset/filtered_image', output_folder='dataset/filtered_image'):
    """
    Обрабатывает все изображения во входной папке со средней интенсивностью >150.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                # Проверка успешной загрузки изображения
                if img is None:
                    print(f"Не удалось загрузить изображение: {filename}")
                    continue

                # Вычисляем среднюю интенсивность как numpy float
                mean_intensity = float(np.mean(img))

                if mean_intensity > 120:
                    print(f"Обработка {filename} (средняя интенсивность: {mean_intensity:.2f})")

                    filtered_img = perona_malik_filter(img)

                    # Нормализуем и сохраняем
                    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

                    # Используем cv2.imwrite для сохранения изображения
                    plt.imsave(output_path, filtered_img, cmap="gray")  # Используем matplotlib для сохранения
                else:
                    print(f"Пропуск {filename} (средняя интенсивность: {mean_intensity:.2f} ≤ 150)")
                    shutil.copy2(input_path, output_path)

            except Exception as e:
                print(f"Ошибка при обработке {filename}: {str(e)}")


if __name__ == "__main__":
    process_images()