import os
import shutil

# СОЗДАНИЕ ДИРЕКТОРИЙ СО ВСЕМИ ИЗОБРАЖЕНИЯМИ И АННОТАЦИЯМИ

source_images_folder = '...' # абсолютный путь к исходникам изображений
source_labels_folder = '...' # абсолютный путь к исходникам аннотаций
destination_images_folder = 'all_images'
destination_labels_folder = 'all_labels'

os.makedirs(destination_images_folder, exist_ok=True)
os.makedirs(destination_labels_folder, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

for root, dirs, files in os.walk(source_images_folder):
    for file in files:
        if file.lower().endswith(image_extensions):
            try:

                src_img_path = os.path.join(root, file)
                dst_img_path = os.path.join(destination_images_folder, file)

                file_label = os.path.splitext(file)[0] + '.txt'
                src_label_path = os.path.join(source_labels_folder, file_label)
                dst_label_path = os.path.join(destination_labels_folder, file_label)

                shutil.copy2(src_img_path, dst_img_path)
                print(f"Изображение скопировано: {src_img_path} -> {dst_img_path}")

                if os.path.exists(src_label_path):
                    shutil.copy2(src_label_path, dst_label_path)
                    print(f"Метка скопирована: {src_label_path} -> {dst_label_path}")
                else:
                    print(f"Внимание: метка не найдена для {file}")

            except Exception as e:
                print(f"Ошибка при обработке файла {file}: {str(e)}")
                continue

print("Копирование завершено!")

