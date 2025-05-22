from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from .database import Base
from sqlalchemy import Column, Integer, VARCHAR, TIMESTAMP, text, LargeBinary, Boolean, ForeignKey


class Images(Base):
    __tablename__ = 'images'

    id = Column(Integer, nullable=False, primary_key=True, index=True)
    filename = Column(VARCHAR(255), nullable=False)
    data = Column(LargeBinary, nullable=False)
    content_type = Column(VARCHAR(100), nullable=False)
    expansion = Column(VARCHAR(255), nullable=False)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))

    detections = relationship("Detections", back_populates="image", cascade='all, delete')

class Detections(Base):
    __tablename__ = 'detected'
    predict_id = Column(Integer, nullable=False, primary_key=True, index=True)
    is_success = Column(Boolean)
    defects = Column(JSONB)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=text('now()'))

    image = relationship("Images", back_populates="detections")