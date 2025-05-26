from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os



# SQLALCHEMY_DB_URL = 'postgresql://postgres:A12138877@localhost/deffection_detect?client_encoding=utf8'  # поменять на env var
SQLALCHEMY_DB_URL = os.getenv("DATABASE_URL",
    "postgresql://postgres:A12138877@localhost/deffection_detect?client_encoding=utf8"
)

engine = create_engine(SQLALCHEMY_DB_URL)

Sessionlocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = Sessionlocal()
    try:
        yield db
    finally:
        db.close()