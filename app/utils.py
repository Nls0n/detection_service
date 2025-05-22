import asyncio
from fastapi import HTTPException
import numpy as np

async def run_visualization_async(image_path: str, weights: str = "last.pt", conf: float = 0.15):
    try:
        process = await asyncio.create_subprocess_exec(
            "python",
            "visualize_predictions.py",
            image_path,
            "--weights", weights,
            "--conf", str(conf),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка выполнения: {stderr.decode()}"
            )

        return stdout.decode()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Системная ошибка: {str(e)}"
        )

def _slice_panorama(img: np.ndarray) -> list[np.ndarray]:
    """Нарезка панорамы на тайлы"""
    SIZE_MAP = {
        (31920, 1152): 28,
        (30780, 1152): 27,
        (18144, 1142): 16,
    }
    h, w = img.shape[:2]
    tiles = SIZE_MAP.get((w, h))
    if tiles is None:
        raise ValueError(f"Неизвестный размер панорамы {w}×{h}")
    tw = w // tiles
    return [img[:, i * tw:(i + 1) * tw] for i in range(tiles)]