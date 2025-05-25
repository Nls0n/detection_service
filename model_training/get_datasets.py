import os
import random
from shutil import copyfile

# ПОЛУЧЕНИЕ СБАЛАНСИРОВАННОГО ДАТАСЕТА

def prepare_dataset(image_dir, label_dir, output_dir, split=0.8):

    for folder in ['images/train', 'images/val',
                   'labels/train', 'labels/val']:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)

    train_end = int(len(image_files) * split)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:]

    for files, split_name in [(train_files, 'train'), (val_files, 'val')]:
        for file in files:

            yolo_label = f"{label_dir}/{file[:-4]}.txt"
            flag = False

            with open(yolo_label, 'r') as file1:
                s = [line.split() for line in file1.readlines()]
                classes = [int(i[0]) for i in s]
                need = [2, 10, 11, 12]
                if sum([1 if i in need else 0 for i in classes]) != 0:
                    flag = True

            if flag:
                src_img = os.path.join(image_dir, file)
                dst_img = os.path.join(output_dir, 'images', split_name, file)
                copyfile(src_img, dst_img)

                label_file = file.replace('.jpg', '.txt')
                src_label = os.path.join(label_dir, label_file)
                dst_label = os.path.join(output_dir, 'labels', split_name, label_file)
                if os.path.exists(src_label):
                    copyfile(src_label, dst_label)
                else:
                    print(f"Warning: Missing label {src_label}")