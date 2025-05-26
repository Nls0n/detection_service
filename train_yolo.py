"""
train_yolo.py — удобная «обёртка» вокруг Ultralytics YOLO v8.

Пример запуска:
    На macbook:
    python train_yolo.py \
        --data data/data.yaml \
        --model yolov8s.pt \
        --epochs 30 \
        --imgsz 960 \
        --batch 4 \
        --device mps \
        --name YOLOv8_M2

    На windows:
    python train_yolo.py `
    --data data/data.yaml `
    --model yolov8s.pt `
    --epochs 30 `
    --imgsz 960 `
    --batch 16 `        # 8 ГБ обычно держит 16-24 в FP16
    --device 0 `
    --half `            # FP16 экономит VRAM и ускоряет
    --workers 8 `       # для Windows максимальное число потоков
    --name YOLOv8_4060


"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def ensure_ultralytics() -> None:
    """
    Проверяем, установлена ли ultralytics;
    если нет — автоматически ставим её в текущий virtualenv.
    """
    if importlib.util.find_spec("ultralytics") is None:
        print("📦 Пакет 'ultralytics' не найден. Устанавливаю ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])
    else:
        import ultralytics as _  # noqa: F401
        print("✅ 'ultralytics' уже установлен.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Training script for YOLO-v8")
    ap.add_argument("--data", type=Path, default=Path("data/data.yaml"),
                    help="Путь к data.yaml")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="Базовая модель или путь к .pt для дообучения")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Сколько эпох обучать")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Размер входных изображений (кратно 32)")
    ap.add_argument("--batch", type=int, default=16,
                    help="Размер батча")
    ap.add_argument("--device", default="0",
                    help="GPU id, 'cpu' или '0,1' для мульти-GPU")
    ap.add_argument("--name", default="exp",
                    help="Имя эксперимента (папка в runs/)")
    ap.add_argument("--resume", action="store_true",
                    help="Продолжить предыдущий run с тем же --name")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_ultralytics()

    from ultralytics import YOLO  # импорт только после проверки

    model = YOLO(args.model)           # загружаем предобученную или последнюю чекпойнт

    # Конвертируем Path → str, чтобы библиотека не «споткнулась»
    data_yaml = str(args.data)

    print("\n🚀 Старт обучения ...\n")
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        resume=args.resume,
        # дополнительные (при желании):
        # workers=8,
        # lr0=0.01,
        # optimizer="SGD",
    )

    print("\n✅ Обучение завершено.")
    print(f"   Лог/чекпойнты: runs/detect/{args.name}")


if __name__ == "__main__":
    main()