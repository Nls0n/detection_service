import os
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, status, HTTPException
from dotenv import load_dotenv
from predict_service.deffect_detector import DefectDetector

load_dotenv()

app = FastAPI()


# model = DefectDetector('weights/best.pt')
HERE = os.path.dirname(__file__)
model_path = os.getenv("MODEL_PATH", os.path.join(HERE, "../app/weights/best.pt"))
model = DefectDetector(model_path)

@app.post("/detect", status_code=status.HTTP_201_CREATED)
async def detect_defects(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        result = model.predict(image)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
    )