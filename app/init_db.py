from database.models import Base
from database.settings import ENGINE

from sqlalchemy import create_engine


# Инициализация базы данных
engine = create_engine(ENGINE)
Base.metadata.create_all(bind=engine)
