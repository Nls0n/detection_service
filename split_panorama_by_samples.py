# split_panorama_by_samples.py

from __future__ import annotations

from pathlib import Path
import cv2
import sys

# --- настройки ---------------------------------------------------------------

SPLITS = ("train", "val", "test")
SRC_FOLDER = Path("data/images")     # корень с origin-изображениями
DST_SUBDIR = "samples"              # куда класть результат

# Карта «размер панорамы  → количество тайлов»
# (width, height) : tiles_in_row
SIZE_MAP: dict[tuple[int, int], int] = {
    (31920, 1152): 28,
    (30780, 1152): 27,
    (18144, 1142): 16,
}

# -----------------------------------------------------------------------------

def slice_panorama(pano_path: Path, out_root: Path) -> None:
    """
    Нарезает панораму согласно SIZE_MAP и сохраняет
    в out_root/<имя панорамы>/<01..N>.png
    """
    img = cv2.imread(str(pano_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"❌ Не удалось открыть {pano_path}")
        return

    h, w = img.shape[:2]
    tiles_in_row = SIZE_MAP.get((w, h))
    if tiles_in_row is None:
        print(f"⚠️  Пропуск {pano_path.name}: неизвестный размер {w}×{h}")
        return

    tile_w = w // tiles_in_row  # ширина одного тайла
    if w % tiles_in_row:
        # теоретически не должно случиться, но проверим
        print(f"⚠️  {pano_path.name}: {w} не делится на {tiles_in_row}")
        return

    pano_name = pano_path.stem
    dest_dir = out_root / pano_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for i in range(tiles_in_row):
        left = i * tile_w
        right = left + tile_w
        tile = img[:, left:right]          # нарезаем по ширине, высоту берём всю
        out_fn = dest_dir / f"{i+1:02d}.png"
        cv2.imwrite(str(out_fn), tile)

    print(f"✓ {pano_path.name}: сохранено {tiles_in_row} сэмплов "
          f"({tile_w}×{h}) в {dest_dir.relative_to(out_root.parent)}")


def process_split(split: str) -> None:
    """
    Обрабатывает один из наборов: train, val или test.
    Ищет панорамы в data/images/<split>/origin/*
    """
    src_dir = SRC_FOLDER / split / "origin"
    dst_root = SRC_FOLDER / split / DST_SUBDIR
    dst_root.mkdir(parents=True, exist_ok=True)

    pano_files = sorted(src_dir.glob("*.png"))
    if not pano_files:
        print(f"⚠️  В {src_dir} не найдено .png-панорам")
        return

    print(f"\n=== {split.upper()} ===  найдено {len(pano_files)} панорам")
    for pano_path in pano_files:
        slice_panorama(pano_path, dst_root)


def main() -> None:
    args = sys.argv[1:]
    if args:
        # пользователь указал конкретные сплиты
        for sp in args:
            if sp not in SPLITS:
                print(f"Неверный split: {sp}. Допустимы {SPLITS}")
                return
            process_split(sp)
    else:
        # обрабатываем все train/val/test
        for sp in SPLITS:
            process_split(sp)


if __name__ == "__main__":
    main()