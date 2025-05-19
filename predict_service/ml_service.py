import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, status, HTTPException
from dotenv import load_dotenv
from predict_service.deffect_detector import DefectDetector
# from .database import engine, get_db, Base

load_dotenv()

app = FastAPI()

# Base.metadata.create_all(bind=engine)

model = DefectDetector('weights/last.pt')


@app.post("/detect", status_code=status.HTTP_201_CREATED)
async def detect_defects(file: UploadFile = File(...)):
    try:
        # Чтение и декодирование изображения
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Предсказание и возврат результата
        result = model.predict(image)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",  # локальный доступ
        port=8080,  # иной порт относительно main.py
    )
