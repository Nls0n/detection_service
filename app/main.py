from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse, HTMLResponse
import app.schemas, app.models
from dotenv import load_dotenv
from app.database import engine, get_db, Base
import cv2
import numpy as np
import threading
from predict_service.ml_service import app as model_app
import uvicorn
from predict_service.ml_service import model

load_dotenv()

application = FastAPI()
application.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)

ml_model = model


# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# mlflow_model_name = os.getenv("MLFLOW_MODEL_NAME")
# mlflow_model_version = 1

# model_name = 'наша модель'
# model_version = 1

@application.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@application.post('/api/predict', status_code=status.HTTP_201_CREATED)
async def predict_deffect(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            content={"message": "File must be an image"})

    content = await file.read()

    db_image = app.models.Images(filename=file.filename, data=content, content_type=file.content_type,
                                 expansion=f'.{file.filename.split('.')[-1]}')
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    last_image = db.query(app.models.Images).order_by(app.models.Images.id.desc()).first()
    image_id = last_image.id if last_image else None
    try:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        raw_data = model.predict(image)

        filtered_data = {
            "status": raw_data["status"],
            "defects": [
                {"class": str(val["class"]), "confidence": f'{float(val["confidence"]) * 100:.2f}%'}
                for val in raw_data["detections"]
            ]
        }
        db_prediction = app.models.Detections(is_success=True, defects=filtered_data["defects"], image_id=image_id)

        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        return filtered_data
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Detection failed {str(e)}')


@application.get('/api/image/{id}', status_code=status.HTTP_200_OK, response_model=app.schemas.GetImage)
def get_image(id: int, db: Session = Depends(get_db)):
    image = db.query(app.models.Images).filter(app.models.Images.id == id).first()
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'image with id {id} not found')

    return image

@application.delete('/api/delete/image/{id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_image(id: int, db: Session = Depends(get_db)):
    image = db.query(app.models.Images).filter(app.models.Images.id == id).first()
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'image with id {id} not found')
    db.delete(image)
    db.commit()


if __name__ == "__main__":
    # Запуск сервиса модели в фоне
    predictor_thread = threading.Thread(
        target=uvicorn.run,
        args=(model_app,),
        kwargs={"host": "0.0.0.0", "port": 8001},
        daemon=True
    )
    predictor_thread.start()

    # Запуск основного API
    uvicorn.run(application, host="0.0.0.0", port=8000)
