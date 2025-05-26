# split_labels_by_samples.py
#
# Делит YOLO-разметку панорамных изображений на N горизонтальных тайлов
# и сохраняет новую разметку в виде отдельных .txt-файлов для каждого тайла.

from __future__ import annotations

import sys
from pathlib import Path

import cv2  # ← OpenCV уже используется в проекте

# ─────────── Настройки  ────────────────────────────────────────────────────
SPLITS = ("train", "val", "test")

ROOT_IMG = Path("data/images")
ROOT_LBL = Path("data/labels")
DST_SUBDIR = "samples"            # куда складываем новые .txt

# (width, height) → количество тайлов по горизонтали
SIZE_MAP: dict[tuple[int, int], int] = {
    (31920, 1152): 28,
    (30780, 1152): 27,
    (18144, 1142): 16,
}
# ───────────────────────────────────────────────────────────────────────────


def clip(v: float, v_min: float, v_max: float) -> float:
    """Ограничиваем значение диапазоном."""
    return max(v_min, min(v, v_max))


def parse_yolo_line(line: str, file_name: str, line_num: int) -> tuple[int, float, float, float, float] | None:
    """
    Разбирает строку YOLO-разметки.
    Если в строке < 5 чисел → предупреждение и пропуск.
    Если ≥ 5 → берём первые 5, остальное игнорируем.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) < 5:
        print(f"⚠️  {file_name}:{line_num}: ожидалось ≥5 чисел, получено {len(parts)}. Строка пропущена.")
        return None

    try:
        cls, xc, yc, w, h = map(float, parts[:5])
    except ValueError as e:
        print(f"⚠️  {file_name}:{line_num}: не удалось преобразовать в float ({e}). Строка пропущена.")
        return None

    return int(cls), xc, yc, w, h


def split_labels(lbl_path: Path,
                 img_size: tuple[int, int],
                 tiles_in_row: int,
                 dest_root: Path) -> None:
    """
    Разрезает один .txt с YOLO-разметкой на tiles_in_row кусочков
    и сохраняет результат в dest_root/<имя_панорамы>/01.txt … NN.txt
    """
    pano_w, pano_h = img_size
    tile_w = pano_w / tiles_in_row               # float, но без остатка
    tile_h = pano_h

    # Буферы строк разметки для каждого тайла
    buffers: list[list[str]] = [[] for _ in range(tiles_in_row)]

    # Читаем и обрабатываем каждую строку
    for idx, raw_line in enumerate(lbl_path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = parse_yolo_line(raw_line, lbl_path.name, idx)
        if parsed is None:
            continue

        cls, xc, yc, w, h = parsed

        # Абсолютные координаты bbox
        box_w = w * pano_w
        box_h = h * pano_h
        cx = xc * pano_w
        cy = yc * pano_h
        x1, y1 = cx - box_w / 2, cy - box_h / 2
        x2, y2 = cx + box_w / 2, cy + box_h / 2

        # Номера тайлов, которые пересекает bbox
        first_tile = int(x1 // tile_w)
        last_tile = int(x2 // tile_w)

        for tile_idx in range(first_tile, last_tile + 1):
            if not (0 <= tile_idx < tiles_in_row):
                # bbox слегка «выглядывает» за панораму — игнорируем часть вне неё
                continue

            t_left = tile_idx * tile_w
            t_right = t_left + tile_w

            # Пересечение bbox с тайлом
            ix1 = clip(x1, t_left, t_right)
            ix2 = clip(x2, t_left, t_right)
            iy1 = clip(y1, 0, tile_h)
            iy2 = clip(y2, 0, tile_h)

            if ix2 <= ix1 or iy2 <= iy1:
                continue  # нет площади пересечения

            # Пересчёт в YOLO-формат для тайла
            new_cx = (ix1 + ix2) / 2 - t_left
            new_cy = (iy1 + iy2) / 2                # вертикальный сдвиг не нужен
            new_w = ix2 - ix1
            new_h = iy2 - iy1

            buffers[tile_idx].append(
                f"{cls} {new_cx / tile_w:.6f} {new_cy / tile_h:.6f} "
                f"{new_w / tile_w:.6f} {new_h / tile_h:.6f}"
            )

    # ───── Сохраняем результат ──────────────────────────────────────────────
    pano_name = lbl_path.stem
    dest_dir = dest_root / pano_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for i, rows in enumerate(buffers, start=1):
        if not rows:
            continue
        (dest_dir / f"{i:02d}.txt").write_text("\n".join(rows))
        created += 1

    print(f"✓ {lbl_path.name}: создано {created} файлов")


def process_split(split: str) -> None:
    """
    Обрабатывает один из наборов (train / val / test).
    """
    src_lbl_dir = ROOT_LBL / split / "origin"
    src_img_dir = ROOT_IMG / split / "origin"
    dst_lbl_root = ROOT_LBL / split / DST_SUBDIR
    dst_lbl_root.mkdir(parents=True, exist_ok=True)

    lbl_files = sorted(src_lbl_dir.glob("*.txt"))
    if not lbl_files:
        print(f"⚠️  В {src_lbl_dir} нет .txt-разметок")
        return

    print(f"\n=== {split.upper()} ===  найдено {len(lbl_files)} файлов разметки")

    for lbl_path in lbl_files:
        img_path = src_img_dir / f"{lbl_path.stem}.png"
        if not img_path.exists():
            print(f"⚠️  {lbl_path.stem}: нет исходного PNG, пропуск")
            continue

        # Определяем размер и количество тайлов
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ Не удалось открыть {img_path}")
            continue

        h, w = img.shape[:2]
        tiles = SIZE_MAP.get((w, h))
        if tiles is None:
            print(f"⚠️  {lbl_path.name}: неизвестный размер {w}×{h}, пропуск")
            continue

        split_labels(lbl_path, (w, h), tiles, dst_lbl_root)


def main() -> None:
    """
    CLI:
        python split_labels_by_samples.py            # обработать train/val/test
        python split_labels_by_samples.py train val  # выборочно
    """
    args = [arg.lower() for arg in sys.argv[1:]]
    if args:
        unknown = [sp for sp in args if sp not in SPLITS]
        if unknown:
            print(f"Неверные split'ы: {unknown}. Допустимы {SPLITS}")
            sys.exit(1)
        splits = args
    else:
        splits = SPLITS

    for sp in splits:
        process_split(sp)


if __name__ == "__main__":
    main()