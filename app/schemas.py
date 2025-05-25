from pydantic import BaseModel
from datetime import datetime

class ExpansionError(BaseModel):
    message: str

class PredictedDefects(BaseModel):
    defects: list[dict]

class SizeError(BaseModel):
    message: str


class UploadImage(BaseModel):
    filename: str
    content_type: str
    data: bytes


class GetImage(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime = datetime.now()

    class DictConfig:
        orm_model = True

class PredictResult(BaseModel):
    status: str
    defects: list[dict[str, str | float]]