from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Модель для сохранения данных
class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    time = Column(String)
    prediction = Column(Float)
    substance = Column(String)
    plant = Column(String)
