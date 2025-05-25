from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse, HTMLResponse
import app.schemas, app.models
from dotenv import load_dotenv
from app.database import engine, get_db, Base
from app.schemas import GetImage
from utils import _slice_panorama, create_defects_report, ndarray_to_bytes
import cv2
from app.models import Images, Detections
import threading
from predict_service.ml_service import app as model_app
import uvicorn
import os
from pathlib import Path
from visualize_predictions import PanoramaProcessor

load_dotenv()

application = FastAPI()


RESULTS_DIR = "static/results"
Path(RESULTS_DIR).mkdir(exist_ok=True)

application.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)

processor = PanoramaProcessor()

@application.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@application.post("/upload")
async def upload_image(file: UploadFile):
    try:
        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)

        temp_path = f"temp_uploads/{file.filename}"
        with open(temp_path, 'wb') as buffer:
            buffer.write(await file.read())

        processor = PanoramaProcessor()

        result_path = processor.process_image(temp_path)



        return {"result_url": f"{result_path}"}

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
    index = 1
    async with httpx.AsyncClient() as client:
        for tile in tiles:
            files = {'file': ('filename.png', ndarray_to_bytes(tile), 'image/png')}

            response = await client.post(
                "http://localhost:8001/detect",
                files=files
            )

            if response.status_code != 201:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ML service error: {response.text}"
                )

            res = response.json()
            processed_tiles.append({"status": res["status"],"defects": [
                    {"class": str(val["class"]), "confidence": f'{float(val["confidence"]) * 100:.2f}%',
                    "index": val["index"], "coordinates": val["coordinates"], "length": val["length"]}
                for val in res["detections"]]})
            index+=1
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
    create_defects_report(processed_tiles)
    return processed_tiles

@application.get('/api/image/{filename}', status_code=status.HTTP_200_OK, response_model=GetImage)
def get_image(filename: str, db: Session = Depends(get_db)):
    image = db.query(Images).filter(Images.filename == filename).first()
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'image with name {filename} not found')

    return image

@application.delete('/api/delete/image/{filename}', status_code=status.HTTP_204_NO_CONTENT)
def delete_image(filename: str, db: Session = Depends(get_db)):
    image = db.query(Images).filter(Images.filename == filename).first()
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'image with id {id} not found')
    db.delete(image)
    db.commit()


@application.get('/report', status_code=status.HTTP_200_OK)
def get_report():
    if not os.path.exists('static/reports/defects_report.docx'):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='report not found')

    return {'report_url': 'static/reports/defects_report.docx'}

if __name__ == "__main__":
    predictor_thread = threading.Thread(
        target=uvicorn.run,
        args=(model_app,),
        kwargs={"host": "0.0.0.0", "port": 8001},
        daemon=True
    )
    predictor_thread.start()

    uvicorn.run(application, host="0.0.0.0", port=8000)
