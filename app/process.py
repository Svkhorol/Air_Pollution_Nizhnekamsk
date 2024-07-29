import os
import pickle
import re

import folium
import numpy as np
import pandas as pd

from database.models import Prediction
from database.settings import ENGINE
from folium.features import DivIcon
from folium.plugins import TimestampedGeoJson
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from predict import (
    extract_value,
    melt_columns,
    concat_dfs
)
from constants import (
    plants_list,
    plants_dict_geojson,
    lags,
    meteo_params,
    substances,
    Mr,
    threshold
)


def extract_coordinates(df):
    """Извлекаются координаты из исходного xlsx-файла"""

    # Строка из df, включающая координаты
    s = list(df)[2]
    s = s.replace(',', '.')
    # Поиск групп цифр, в том числе разделенных точкой
    pattern = r'[-+]?\d*\.\d+|\d+'
    matches = re.findall(pattern, s)
    coordinates = [float(match) for match in matches if '.' in match and len(match) >= 9]
    # Добавление в список постов контроля воздуха
    locations_list.append(coordinates)


# Функция загрузки и подготовки данных по нужным веществам из всех 6 геоточек
def load_data(file, all_substances, meteo_params):
    """file - файл с данными для определенной геоточки
       all_substances - названия загрязняющих веществ
    """

    point = pd.read_excel(os.path.join(path, file))

    # Сохранение координат геоточки
    extract_coordinates(point)

    # Обработка индексов строк и столбцов
    indexes = list(point.loc[0])
    indexes = indexes[1:]
    point = point.loc[3:].set_index('Интервал отбора')
    point.index = pd.to_datetime(point.index, format='%d.%m.%Y %H:%M')
    point.index = point.index.strftime('%Y-%m-%d %H:%M')
    point.columns = indexes

    point = point.apply(lambda x: x.str.replace(',', '.'))
    point['V ветра, м/с'] = point['V ветра, м/с'].apply(extract_value)
    point[['Угол ветра, °', 'Направление ветра']] = point['D ветра, °'].str.extract(r'(\d+)\s?\((.+)\)')
    point['Направление ветра'].replace(np.NaN, 'Ш', inplace=True)

    # Пересечение множеств названий столбцов и substances
    substances = set(indexes) & set(all_substances)
    columns = [*substances, *meteo_params]
    point = point[columns]

    # Заполнение пропусков, если их в столбце не более 5%
    for col in point.columns:
        point[col] = pd.to_numeric(point[col], errors='ignore')
        if point[col].isna().sum().sum() / len(point) < 0.05:
            point[col].interpolate(inplace=True)
    # Заполнение всех пропусков для Угла ветра (относятся к "Ш")
    point['Угол ветра, °'].interpolate(inplace=True)

    return point


def save_to_sqlite(series: pd.Series, substance: str, plant: str) -> None:
    """Сохранение данных в базу данных"""

    data_to_db = pd.DataFrame({'time': series.index, 'prediction': series})
    data_to_db['substance'] = substance
    data_to_db['plant'] = plant
    # Преобразование в список словарей
    db_dicts = data_to_db.to_dict(orient='records')
    session.bulk_insert_mappings(Prediction, db_dicts)
    session.commit()


# Границы концентраций для различных веществ
def get_pollut_coor(prediction: float):
    if prediction <= threshold[substance][0]:
        return [0, -90]  # южный полюс
    else:
        return coors  # предприятие


def get_threshold_coor(prediction: float):
    if prediction < threshold[substance][1]:
        return [0, -90]  # южный полюс
    else:
        return coors  # предприятие


def popup(idx, row):
    popup = f"""
        <p style='color:blue;'>
            {idx}                         <!-- время -->
        </p>      
        <h2>{substance}</h2>              <!-- вещество -->
        <h3>
        <code>
            {row['prediction']} мг/м3     <!-- концентрация -->
        </code>
        </h3>
        <p> {plant} </p>                  <!-- предприятие -->
        {[coors[1], coors[0]]}            <!-- координаты -->
        """
    return popup


def get_pollut_start():
    indices = data[data['prediction'] >= pollut_level].index
    if not indices.empty:
        # Время первого загрязнения
        pollution[0].append(indices[0])
        # Место первого загрязнения
        pollution[1].append(plant)


# Инфо на карте о месте появления выбросов
class MapText:
    def __init__(self, pollution):
        self.title = """
            <div
                style="font-size: 10pt;
                       color: gray;
                       background: white">
                <b>Первое загрязнение воздуха:</b>
            </div>
        """
        self.text_template = """
            <div
                style="font-size: 9pt;
                       color: gray;
                       background: white">
                {}
            </div>
        """
        self.pollution = pollution

    def generate_message(self):
        if self.pollution[0]:
            text = [
                self.text_template.format(f"{time}: {plant}")
                for time, plant in zip(self.pollution[0], self.pollution[1])
            ]
        else:
            text = [self.text_template.format("Содержание загрязняющих веществ не превышалось")]

        message = "\n".join([self.title] + text)
        return message

    def add_text_to_map(self, map_obj):
        message = self.generate_message()
        folium.map.Marker(
            [55.660473, 51.794859],
            icon=DivIcon(
                icon_size=(250, 36),
                html=message,
            )
        ).add_to(map_obj)


# Директория с данными
path = '../data/raw/test'
files = os.listdir(path)

# Настройки подкючения к sqlite
engine = create_engine(ENGINE)
Session = sessionmaker(bind=engine)
session = Session()

# Пункты контроля воздуха
locations_list = []

# Список из входных датасетов для каждого предприятия
X_list = []

# Загрузка и обработка входных данных
points = [load_data(file, substances, meteo_params) for file in files if file.endswith('.xlsx')]
df_melted_list = [melt_columns(df, substances, Mr, meteo_params) for df in points]

# Вычисляем расстояния из всех контрольных пунктов до предприятий
for target_num in range(len(plants_list)):
    X = concat_dfs(df_melted_list, locations_list, target_num + 1, lags)
    X.dropna(inplace=True)
    X = pd.get_dummies(X, columns=['Направление ветра'])
    X_list.append(X)

# Сохранение предсказаний на карте и в базе данных
for substance in substances:

    features = []  # cвойства объектов GeoJson
    pollution = [[], []]  # [[time], [plant]]
    pollut_level = threshold[substance][0]

    # Загрузка моделей для определения каждого вещества
    with open(f'ml_models/model_{substance[:3]}.pkl', 'rb') as file:
        catboost = pickle.load(file)

    for X, (plant, coors) in zip(X_list, plants_dict_geojson.items()):
        pred = catboost.predict(X)

        # Замена отрицательных значений
        pred[pred < 0] = 0
        # Сортировка значений по времени
        pred = pd.Series(pred, index=X.index)
        pred_sorted = pred.sort_index()
        # Максимальные из предсказанных значений
        max_pred = pred_sorted.groupby(level=0).max()

        # Сохранение данных в базу данных
        save_to_sqlite(max_pred)

        # Подговка данных для отображения на карте
        data = pd.DataFrame({'prediction': max_pred.round(4)})
        # Параметры маркера на карте
        data['pollut_coor'] = data['prediction'].apply(get_pollut_coor)
        data['threshold_coor'] = data['prediction'].apply(get_threshold_coor)

        # Индексы времени с повышенной концентрацией
        # Заполнение списка pollution
        get_pollut_start()

        # Всплывающее окно с информацией
        for idx, row in data.iterrows():
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": coors
                    },
                    "properties": {
                        "time": idx,
                        "popup": popup(idx, row),
                        "icon": "marker",
                    },
                }
            )

        # Сигнал о повышении концентрации
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": list(data['pollut_coor']),
                },
                "properties": {
                    "times": list(data.index),
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": "red",
                        "fillOpacity": 0.3,
                        "radius": 15,
                    },
                    "style": {"weight": 0},
                },
            }
        )

        # Сигнал о превышении ПДК
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": list(data['threshold_coor']),
                },
                "properties": {
                    "times": list(data.index),
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": "red",
                        "fillOpacity": 0.5,
                        "radius": 18,
                    },
                    "style": {"weight": 0},
                },
            }
        )
    map_substance = folium.Map(location=[55.596030, 51.916050], zoom_start=11.5)

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT30M",  # через 30 мин
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options="YYYY-MM-DD HH:mm",
        time_slider_drag_update=True,
    ).add_to(map_substance)

    # Текстовый заголовок
    message = MapText(pollution)
    message.add_text_to_map(map_substance)

    # Сохранение карты для каждого вещества
    map_substance.save(f"templates/processed_map_{substance[:3]}.html")

session.close()