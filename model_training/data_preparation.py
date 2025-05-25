import os
import cv2
import numpy as np
from tqdm import tqdm

# ПОДГОТОВКА ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ,
# ПРОПУСКАЮТСЯ ИЗОБРАЖЕНИЯ С СО СРЕДНЕЙ ИНТЕНСИВНОСТЬЮ >= 170

def process_panoramas(input_images_dir='all_images/',
                      input_labels_dir='all_labels/',
                      output_images_dir='datasetz/images/',
                      output_labels_dir='datasetz/labels/',
                      tile_width=1140,
                      tile_height=1152,
                      intensity_threshold=170):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Обработка панорам"):
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(input_images_dir, image_file)
        label_path = os.path.join(input_labels_dir, f"{base_name}.txt")

        image = cv2.imread(image_path)
        if image is None:
            print(f"\nВнимание: не прочитан файл изображения {image_path}\n")
            continue

        H, W = image.shape[:2]

        if H != tile_height:
            print(f"\nВнимание:  высота изображения {H} не соответствует {tile_height} для {image_file}\n")
            continue

        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                annotations = [line.strip().split() for line in f.readlines() if line.strip()]

        if W % tile_width != 0:
            print(f"\nВнимание: отсутствует целое число тайлов ({tile_width}x{tile_height}) в {image_file}\n")
            continue

        for col in range(W // tile_width):
            x_start = col * tile_width
            x_end = x_start + tile_width

            tile = image[0:tile_height, x_start:x_end]

            gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_tile)

            if mean_intensity >= intensity_threshold:
                continue

            tile_annotations = []
            for ann in annotations:
                class_id = ann[0]
                points = list(map(float, ann[1:]))

                abs_points = [(x * W, y * H) for x, y in zip(points[::2], points[1::2])]

                x_min = min(x for x, y in abs_points)
                x_max = max(x for x, y in abs_points)

                if x_max < x_start or x_min > x_end:
                    continue

                new_points = []
                for x, y in abs_points:
                    x_new = max(x_start, min(x, x_end))
                    y_new = y
                    x_rel = (x_new - x_start) / tile_width
                    y_rel = y_new / tile_height
                    new_points.extend([x_rel, y_rel])

                if len(new_points) >= 4:
                    tile_annotations.append(f"{class_id} " + " ".join(map(str, new_points)))

            tile_filename = f"{base_name}_{col}.jpg"
            label_filename = f"{base_name}_{col}.txt"

            cv2.imwrite(os.path.join(output_images_dir, tile_filename), tile)

            if tile_annotations:
                with open(os.path.join(output_labels_dir, label_filename), 'w') as f:
                    f.write("\n".join(tile_annotations))
            else:
                open(os.path.join(output_labels_dir, label_filename), 'w').close()


if __name__ == "__main__":
    process_panoramas()