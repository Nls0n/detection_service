#!/usr/bin/env python3
"""
Расширенная версия: помимо замены префикса умеет
генерировать пути к сэмплам вместо origin-кадров.

Пример:
    python update_sample_lists.py \
        --lists-dir data \
        --old-root images \
        --new-root images      # префикс сохраняем
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List


def collect_sample_paths(origin_path: Path,
                         origin_dir_name: str = "origin",
                         samples_dir_name: str = "samples") -> List[Path]:
    """
    origin_path: .../<split>/origin/<file>.png
    вернёт      : список .../<split>/samples/<file>/<*.png>
    """
    if origin_dir_name not in origin_path.parts:
        return []  # на всякий — строка не из origin
    # index 'origin' в пути
    idx = origin_path.parts.index(origin_dir_name)
    # /.../<split>/samples/<file-stem>/
    sample_dir = Path(*origin_path.parts[:idx]) / samples_dir_name / origin_path.stem
    # glob внутри каталога
    return sorted(sample_dir.glob("*.png"))


def process_txt(txt_path: Path,
                old_root: str,
                new_root: str) -> List[str]:
    lines_out: List[str] = []
    for raw in txt_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue

        # 1) меняем префикс, если совпал
        if raw.startswith(old_root):
            raw = new_root + raw[len(old_root):]

        origin_p = Path(raw)
        # 2) расширяем на samples
        samples = collect_sample_paths(origin_p)
        if samples:
            lines_out.extend(map(str, samples))
        else:
            # если samples не найдены, всё равно пишем (чтобы не терять)
            lines_out.append(str(origin_p))
    return lines_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists-dir", default=".", type=Path,
                    help="где лежат train.txt / val.txt / test.txt")
    ap.add_argument("--old-root", required=True,
                    help="что заменить в начале пути")
    ap.add_argument("--new-root", required=True,
                    help="чем заменить old-root")
    args = ap.parse_args()

    for name in ("train", "val", "test"):
        src = args.lists_dir / f"{name}.txt"
        if not src.exists():
            continue
        dst = args.lists_dir / f"{name}_samples.txt"
        new_lines = process_txt(src, args.old_root, args.new_root)
        dst.write_text("\n".join(new_lines))
        print(f"{dst}  ←  {len(new_lines)} строк")


if __name__ == "__main__":
    main()