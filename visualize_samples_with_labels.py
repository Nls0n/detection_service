# visualize_samples_with_labels.py
# YOLOv8 → нарезка панорамы → предсказания → склейка и показ
'''
пример
python visualize_samples_with_labels.py data/images/val/samples/0-691-ls-14-d03   --weights last.pt --conf 0.15
'''

from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import yaml
from matplotlib import font_manager as fm

# ─── карта «размер панорамы → количество тайлов» ───────────────────────────
SIZE_MAP = {
    (31920, 1152): 28,
    (30780, 1152): 27,
    (18144, 1142): 16,
}
# ───────────────────────────────────────────────────────────────────────────

# ─── попытка подобрать шрифт с поддержкой Юникода ──────────────────────────
def pick_font(size: int) -> ImageFont.ImageFont:
    try:
        path = fm.findfont("DejaVu Sans")          # Matplotlib ставит DejaVu Sans
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

FONT_SIZE = 14
FONT = pick_font(FONT_SIZE)

# ─── утилиты ───────────────────────────────────────────────────────────────
def load_class_names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data["names"].items()}

def get_text_size(draw: ImageDraw.ImageDraw, txt: str) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):                # Pillow ≥ 8
        x0, y0, x1, y1 = draw.textbbox((0, 0), txt, font=FONT)
        return x1 - x0, y1 - y0
    return FONT.getsize(txt)                     # старые Pillow

def slice_panorama(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    tiles = SIZE_MAP.get((w, h))
    if tiles is None:
        raise SystemExit(f"Неизвестный размер панорамы {w}×{h}")
    tw = w // tiles
    return [img[:, i * tw:(i + 1) * tw] for i in range(tiles)]

def join_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(tiles, axis=1)

# ─── отрисовка результата на одном тайле ───────────────────────────────────
def draw_preds(tile_bgr: np.ndarray,
               res,
               names: dict[int, str],
               conf_th: float) -> np.ndarray:
    """Возвращает RGB-тайл с нарисованными контурами/боксами и подписями."""
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    drw = ImageDraw.Draw(pil, "RGBA")

    have_masks = res.masks is not None and len(res.masks.xy) > 0

    # --- либо маски (контуры), либо боксы ----------------------------------
    if have_masks:                                      # моделb-segmentation
        for box, poly in zip(res.boxes, res.masks.xy):
            conf = float(box.conf[0])
            if conf < conf_th:
                continue
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id))

            # контур
            pts = [(float(x), float(y)) for x, y in poly]
            drw.line(pts + [pts[0]], fill=(0, 255, 0, 255), width=2)

            # подпись
            tw, th = get_text_size(drw, label)
            x0, y0 = pts[0]
            drw.rectangle([x0, y0 - th - 2, x0 + tw + 4, y0],
                           fill=(0, 0, 0, 90))
            drw.text((x0 + 2, y0 - th - 1), label,
                     font=FONT, fill=(255, 255, 255, 255))
    else:                                               # только боксы
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < conf_th:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cls_id = int(box.cls[0])
            label  = names.get(cls_id, str(cls_id))

            # рамка
            drw.rectangle([x1, y1, x2, y2],
                           outline=(255, 0, 0, 255), width=2)

            # подпись
            tw, th = get_text_size(drw, label)
            drw.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1],
                           fill=(0, 0, 0, 90))
            drw.text((x1 + 2, y1 - th - 1), label,
                     font=FONT, fill=(255, 255, 255, 255))

    return np.asarray(pil)

# ─── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        "Визуализатор предсказаний YOLOv8 на панорамах / тайлах")
    p.add_argument("input",
                   type=Path,
                   help="Панорама PNG ИЛИ директория с тайлами *.png")
    p.add_argument("--weights", type=Path, default="last.pt",
                   help="YOLOv8 weights (default: last.pt)")
    p.add_argument("--yaml", type=Path, default="data/data.yaml",
                   help="data.yaml с названиями классов")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence-threshold (default 0.25)")
    return p.parse_args()

# ─── main ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    names = load_class_names(args.yaml)
    model = YOLO(str(args.weights))

    # —  получаем список BGR-тайлов  —
    if args.input.is_dir():
        paths = sorted(args.input.glob("*.png"))
        if not paths:
            raise SystemExit(f"Нет .png в {args.input}")
        tiles_bgr = [cv2.imread(str(p)) for p in paths]
        out_path  = None
    else:
        pano_bgr = cv2.imread(str(args.input))
        if pano_bgr is None:
            raise SystemExit(f"Не удалось открыть {args.input}")
        tiles_bgr = slice_panorama(pano_bgr)
        out_path  = args.input.with_name(args.input.stem + "_pred.png")

    # —  инференс + отрисовка на каждом тайле  —
    processed = []
    for tile in tiles_bgr:
        res = model.predict(tile, conf=args.conf, verbose=False)[0]
        processed.append(
            draw_preds(tile, res, names, args.conf)
        )

    # —  склейка и показ  —
    pano_rgb = join_tiles(processed)
    h, w = pano_rgb.shape[:2]
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    plt.imshow(pano_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # —  сохранение результата, если исходник был одной панорамой  —
    if out_path:
        cv2.imwrite(str(out_path), cv2.cvtColor(pano_rgb, cv2.COLOR_RGB2BGR))
        print(f"✓ Результат сохранён: {out_path}")

# ────
if __name__ == "__main__":
    main()