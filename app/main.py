from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse, HTMLResponse
import app.schemas, app.models
from dotenv import load_dotenv
from app.database import engine, get_db, Base
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictResult
from utils import _slice_panorama
import cv2
from app.models import Images, Detections
import numpy as np
import threading
from predict_service.ml_service import app as model_app
import uvicorn
from predict_service.ml_service import model
import os
from pathlib import Path
from visualize_predictions import PanoramaProcessor
import uuid
load_dotenv()

application = FastAPI()


RESULTS_DIR = "static/results"
Path(RESULTS_DIR).mkdir(exist_ok=True)

application.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)

ml_model = model
processor = PanoramaProcessor()

@application.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@application.post("/upload")
async def upload_image(file: UploadFile):
    # Сохраняем загруженный файл
    try:
        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)

        filename = f"processed_{file.filename}"
        output_path = f"static/results/{filename}"
        temp_path = f"temp_uploads/{file.filename}"
        with open(temp_path, 'wb') as buffer:
            buffer.write(await file.read())

        processor = PanoramaProcessor()

        # Самый простой вариант вызова - только путь к изображению
        result_path = processor.process_image(temp_path)



        return {"result_url": f"/static/results/{filename}"}

    except Exception as e:
        return {"error": str(e)}




@application.post('/api/predict', status_code=status.HTTP_201_CREATED)
async def predict_deffect(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            content={"message": "File must be an image"})

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    temp_path = f"temp_uploads/{file.filename}"
    with open(temp_path, 'wb') as buffer:
        buffer.write(content)

    img = cv2.imread(temp_path)
    if img is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Could not read image file")

    tiles = _slice_panorama(img)
    processed_tiles = []
    for tile in tiles:
        res = model.predict(tile)
        processed_tiles.append({"status": res["status"],"defects": [
                {"class": str(val["class"]), "confidence": f'{float(val["confidence"]) * 100:.2f}%'}
                for val in res["detections"]]})
    db_image = Images(filename=file.filename, data=content, content_type=file.content_type,
                                 expansion=f'.{file.filename.split('.')[-1]}')
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    last_image = db.query(app.models.Images).order_by(app.models.Images.id.desc()).first()
    db_predictions = Detections(is_success=True, defects=processed_tiles, image_id=last_image.id)
    db.add(db_predictions)
    db.commit()
    db.refresh(db_predictions)
    return processed_tiles

    # image_id = last_image.id if last_image else None
    # try:
    #     nparr = np.frombuffer(content, np.uint8)
    #     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #     raw_data = model.predict(image)
    #
    #     filtered_data = {
    #         "status": raw_data["status"],
    #         "defects": [
    #             {"class": str(val["class"]), "confidence": f'{float(val["confidence"]) * 100:.2f}%'}
    #             for val in raw_data["detections"]
    #         ]
    #     }
    #     db_prediction = app.models.Detections(is_success=True, defects=filtered_data["defects"], image_id=image_id)
    #
    #     db.add(db_prediction)
    #     db.commit()
    #     db.refresh(db_prediction)
    #
    #     return filtered_data
    # except Exception as e:
    #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Detection failed {str(e)}')


@application.get('/api/image/{id}', status_code=status.HTTP_200_OK, response_model=app.schemas.GetImage)
def get_image(id: int, db: Session = Depends(get_db)):
    image = db.query(Images).filter(Images.id == id).first()
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'image with id {id} not found')

    return image

@application.delete('/api/delete/image/{id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_image(id: int, db: Session = Depends(get_db)):
    image = db.query(Images).filter(Images.id == id).first()
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'image with id {id} not found')
    db.delete(image)
    db.commit()


if __name__ == "__main__":
    # Запуск сервиса модели в другом потоке
    predictor_thread = threading.Thread(
        target=uvicorn.run,
        args=(model_app,),
        kwargs={"host": "0.0.0.0", "port": 8001},
        daemon=True
    )
    predictor_thread.start()

    # Запуск основного API
    uvicorn.run(application, host="0.0.0.0", port=8000)
