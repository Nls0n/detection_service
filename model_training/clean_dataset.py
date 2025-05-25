import os
import glob
import re

# УДАЛЕНИЕ ИЗОБРАЖЕНИЙ С ПУСТЫМИ АННОТАЦИЯМИ

def rename_empty_annotation_pairs(images_dir, annotations_dir):

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    annotation_extensions = ['*.txt', '*.xml', '*.json']

    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    annotation_dict = {}
    for ext in annotation_extensions:
        for ann_path in glob.glob(os.path.join(annotations_dir, ext)):
            ann_name = os.path.splitext(os.path.basename(ann_path))[0]
            annotation_dict[ann_name] = ann_path

    counter = 1

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_ext = os.path.splitext(img_path)[1]

        if img_name in annotation_dict:
            ann_path = annotation_dict[img_name]

            if os.path.getsize(ann_path) == 0:

                new_name = f"incorrect_{counter}"

                new_img_path = os.path.join(images_dir, new_name + img_ext)
                os.rename(img_path, new_img_path)

                ann_ext = os.path.splitext(ann_path)[1]
                new_ann_path = os.path.join(annotations_dir, new_name + ann_ext)
                os.rename(ann_path, new_ann_path)

                counter += 1
                print(f"Renamed: {img_name} to {new_name}")

        else:

            new_name = f"incorrect_{counter}"

            new_img_path = os.path.join(images_dir, new_name + img_ext)
            os.rename(img_path, new_img_path)

            counter += 1
            print(f"Renamed: {img_name} to {new_name}")

def delete_incorrect_files(directory):

    pattern = re.compile(r'incorrect_\d+.*')

    deleted_count = 0

    for filename in os.listdir(directory):
        if pattern.fullmatch(filename):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Удалён: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Ошибка при удалении {file_path}: {e}")

    print(f"Всего удалено файлов: {deleted_count}")


images_directory = "..."
labels_directory = "..."

rename_empty_annotation_pairs(images_directory, labels_directory)
delete_incorrect_files(images_directory)
delete_incorrect_files(labels_directory)