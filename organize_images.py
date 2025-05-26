# organize_images.py

import shutil
import random
from pathlib import Path

def copy_split(txt_path: Path, dest_dir: Path, file_map: dict):
    """
    Копирует файлы из списка txt_path в папку dest_dir, используя заранее
    построенный file_map (имя→полный путь).
    """
    if not txt_path.exists():
        print(f'ERROR: не найден файл {txt_path}')
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with txt_path.open('r', encoding='utf-8') as f:
        for line in f:
            filename = Path(line.strip()).name
            if not filename:
                continue
            src = file_map.get(filename)
            if src:
                dst = dest_dir / filename
                shutil.copy(src, dst)
                count += 1
                print(f'Copied to {dest_dir.name}: {filename}')
            else:
                print(f'WARNING: {filename} не найден в исходных данных')
    return count

def clean_split(dest_dir: Path, labels_dir: Path):
    """
    Удаляет из dest_dir все PNG-файлы, для которых нет соответствующего
    .txt в labels_dir.
    """
    if not labels_dir or not labels_dir.exists():
        return 0

    deleted = 0
    for img in dest_dir.glob('*.png'):
        label = labels_dir / f'{img.stem}.txt'
        if not label.exists():
            img.unlink()
            deleted += 1
            print(f'Deleted (no label): {img.name}')
    return deleted

def split_val_set(
    train_img_dir: Path,
    train_lbl_dir: Path,
    val_img_dir: Path,
    val_lbl_dir: Path,
    val_txt: Path,
    train_txt: Path,
    ratio: float = 0.1
):
    """
    Выделяет ratio долю изображений вместе с их .txt в папку val,
    создаёт val.txt и train.txt с путями к новым наборам.
    """
    imgs = list(train_img_dir.glob('*.png'))
    n_val = max(1, int(len(imgs) * ratio))
    val_imgs = set(random.sample(imgs, n_val))

    # Создаём папки
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Перемещаем и записываем val.txt
    with val_txt.open('w', encoding='utf-8') as vf:
        for img in val_imgs:
            # move image
            dst_img = val_img_dir / img.name
            shutil.move(str(img), str(dst_img))
            # move label if есть
            lbl_src = train_lbl_dir / f'{img.stem}.txt'
            if lbl_src.exists():
                dst_lbl = val_lbl_dir / lbl_src.name
                shutil.move(str(lbl_src), str(dst_lbl))
            # записываем путь в val.txt
            vf.write(f'{val_img_dir.as_posix()}/{img.name}\n')

    print(f'Moved {n_val} images + labels to {val_img_dir.name} / {val_lbl_dir.name}')

    # Записываем train.txt из оставшихся
    remaining = list(train_img_dir.glob('*.png'))
    with train_txt.open('w', encoding='utf-8') as tf:
        for img in remaining:
            tf.write(f'{train_img_dir.as_posix()}/{img.name}\n')
    print(f'Wrote {len(remaining)} remaining images to {train_txt}')

def main():
    source_dir = Path('data/films-1000')
    splits = {
        'train': {
            'txt':    Path('data/train_base.txt'),
            'img':    Path('data/images/train/origin'),
            'lbl':    Path('data/labels/train/origin'),
        },
        'test': {
            'txt':    Path('data/test.txt'),
            'img':    Path('data/images/test/origin'),
            'lbl':    None,
        }
    }

    # 1. Сбор .png и копирование train/test
    print(f'Сканирование {source_dir}…')
    file_map = {p.name: p for p in source_dir.rglob('*.png')}

    for name, cfg in splits.items():
        print(f'\n=== Обработка "{name}" ===')
        copied = copy_split(cfg['txt'], cfg['img'], file_map)
        deleted = clean_split(cfg['img'], cfg['lbl'])
        print(f'{name}: скопировано {copied}, удалено {deleted}')

    # 2. Выделение валидации из train
    print('\n=== Выделение 10% валидации ===')
    split_val_set(
        train_img_dir=Path('data/images/train/origin'),
        train_lbl_dir=Path('data/labels/train/origin'),
        val_img_dir=Path('data/images/val/origin'),
        val_lbl_dir=Path('data/labels/val/origin'),
        val_txt=Path('data/val.txt'),
        train_txt=Path('data/train.txt'),
        ratio=0.1
    )

if __name__ == '__main__':
    main()