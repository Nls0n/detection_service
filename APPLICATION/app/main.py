# APPLICATION/app/main.py

from pathlib import Path
import os
# import threading

import cv2
import httpx
# import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

import app.schemas as schemas
from app.schemas import GetImage, PredictResult
from app.models import Images, Detections
from app.database import engine, get_db, Base
from app.utils import _slice_panorama, create_defects_report, ndarray_to_bytes
from app.visualize_predictions import PanoramaProcessor
# from predict_service.ml_service import app as model_app

# -----------------------------------------------------------------------------
# Загрузка переменных окружения
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# Конфигурация путей к директориям
# -----------------------------------------------------------------------------
HERE      = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
STATIC    = HERE / "static"
RESULTS   = STATIC / "results"
REPORTS   = STATIC / "reports"

# Убеждаемся, что нужные директории существуют
RESULTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Инициализация FastAPI-приложения
# -----------------------------------------------------------------------------
application = FastAPI(title="AI Weld Analysis Frontend")

# Разрешаем CORS для любых источников (для разработки)
application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем статические файлы и настраиваем шаблоны
application.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES))

# Создаем таблицы в БД при старте
Base.metadata.create_all(bind=engine)

# Создаем экземпляр PanoramaProcessor для визуализации
processor = PanoramaProcessor()

# Адрес ML-сервиса
ML_SERVICE_BASE_URL  = os.getenv("ML_SERVICE_URL", "http://localhost:8001")
ML_SERVICE_DETECT_EP = f"{ML_SERVICE_BASE_URL}/detect"


@application.get("/", response_class=HTMLResponse)
def read_root(request: Request) -> HTMLResponse:
    """
    Отобразить главную страницу с интерфейсом загрузки и просмотра результатов.

    Args:
        request (Request): Объект запроса FastAPI.

    Returns:
        HTMLResponse: Отрендеренный шаблон index.html.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@application.post("/upload")
async def upload_image(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Сохранить загруженную панораму и обработать ее PanoramaProcessor.

    Args:
        file (UploadFile): Загруженный файл изображения.

    Returns:
        dict: Словарь с ключом 'result_url' — относительный путь к обработанному изображению.

    Raises:
        HTTPException: При ошибке чтения или обработки файла.
    """
    try:
        temp_dir  = HERE / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Пустой файл")

        temp_path.write_bytes(content)

        # output_path = processor.process_image(str(temp_path))
        output_path, metadata = processor.process_image(str(temp_path))
        filename    = Path(output_path).name
        return {"result_url": f"/static/results/{filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@application.post(
    "/api/predict",
    status_code=status.HTTP_201_CREATED,
    response_model=list[PredictResult]
)
async def predict_defect(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> list[dict]:
    """
    Разрезать панораму на тайлы, отправить каждый тайл в ML-сервис,
    сохранить результаты в БД и сформировать отчёт.

    Args:
        file (UploadFile): Загруженный файл панорамы.
        db (Session): Сессия SQLAlchemy для работы с БД.

    Returns:
        list[dict]: Список словарей для каждого тайла:
            - status: "success" или "no_defects"
            - defects: список дефектов с полями:
                class, confidence, index, coordinates, length
    """
    # Проверка типа файла
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"message": "Файл должен быть изображением"}
        )

    # Считываем содержимое
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Пустой файл")

    # Временное сохранение для OpenCV
    temp_dir  = HERE / "temp_uploads"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename
    temp_path.write_bytes(content)

    # Декодируем изображение
    img = cv2.imread(str(temp_path))
    if img is None:
        raise HTTPException(status_code=422, detail="Не удалось прочитать изображение")

    # Разбиваем панораму на тайлы
    tiles   = _slice_panorama(img)
    results = []

    # Отправка запросов к ML-сервису
    async with httpx.AsyncClient(timeout=60.0) as client:
        for tile in tiles:
            payload = {
                'file': (
                    'tile.png',
                    ndarray_to_bytes(tile, format="png"),
                    'image/png'
                )
            }
            resp = await client.post(ML_SERVICE_DETECT_EP, files=payload)
            if resp.status_code != status.HTTP_201_CREATED:
                raise HTTPException(
                    status_code=502,
                    detail=f"Ошибка ML-сервиса: {resp.text}"
                )
            ml_data = resp.json()

            # Формируем запись с необходимыми полями
            results.append({
                "status": ml_data.get("status"),
                "defects": [
                    {
                        "class":       d["class"],
                        "confidence":  f"{d['confidence']*100:.2f}%",
                        "index":       d["index"],
                        "coordinates": d["coordinates"],
                        "length":      d["length"],
                    }
                    for d in ml_data.get("detections", [])
                ]
            })

    # Сохраняем изображение в БД
    db_image = Images(
        filename=file.filename,
        data=content,
        content_type=file.content_type,
        expansion=f".{file.filename.split('.')[-1]}"
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    # Сохраняем детекции в БД
    db_pred = Detections(
        is_success=any(r["status"] == "success" for r in results),
        defects=results,
        image_id=db_image.id
    )
    db.add(db_pred)
    db.commit()

    # Генерируем отчет Word
    # create_defects_report(results)
    create_defects_report(
        results,
        output_filename=str(REPORTS / "defects_report.docx")
    )

    return results


@application.get(
    "/api/image/{filename}",
    response_model=GetImage,
    status_code=status.HTTP_200_OK
)
def get_image(filename: str, db: Session = Depends(get_db)) -> GetImage:
    """
    Получить информацию о сохраненном изображении по его имени.

    Args:
        filename (str): Имя файла в БД.
        db (Session): Сессия SQLAlchemy.

    Returns:
        GetImage: Pydantic-модель с id, filename, uploaded_at.
    """
    image = db.query(Images).filter(Images.filename == filename).first()
    if not image:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return image


@application.delete(
    "/api/delete/image/{filename}",
    status_code=status.HTTP_204_NO_CONTENT
)
def delete_image(filename: str, db: Session = Depends(get_db)) -> None:
    """
    Удалить запись об изображении и все связанные детекции по имени файла.

    Args:
        filename (str): Имя файла для удаления.
        db (Session): Сессия SQLAlchemy.
    """
    image = db.query(Images).filter(Images.filename == filename).first()
    if not image:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    db.delete(image)
    db.commit()


@application.get("/report", status_code=status.HTTP_200_OK)
def get_report() -> dict[str, str]:
    """
    Вернуть ссылку на сгенерированный Word-отчет, если он существует.

    Returns:
        dict: {'report_url': '/static/reports/defects_report.docx'}

    Raises:
        HTTPException: Если файл отчета не найден.
    """
    report_path = REPORTS / "defects_report.docx"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Отчет не найден")
    return {"report_url": f"/static/reports/{report_path.name}"}


# -----------------------------------------------------------------------------
# Для локального запуска обоих сервисов в одном процессе
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Запускаем ML-сервис на порте 8001 в фоновом потоке
#     predictor_thread = threading.Thread(
#         target=uvicorn.run,
#         args=(model_app,),
#         kwargs={"host": "0.0.0.0", "port": 8001, "log_level": "info"},
#         daemon=True
#     )
#     predictor_thread.start()
#
#     # Запускаем фронтенд на порте 8000
#     uvicorn.run(application, host="0.0.0.0", port=8000, log_level="info")