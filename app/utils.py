import asyncio
from fastapi import HTTPException


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