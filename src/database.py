import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, TIMESTAMP
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    sex = Column(String(10), nullable=False)
    children = Column(Integer, nullable=False)
    smoker = Column(String(10), nullable=False)
    region = Column(String(20), nullable=False)
    predicted_cost = Column(Float, nullable=False)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)
