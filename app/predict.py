import os
import pickle
import re

import folium
import numpy as np
import pandas as pd

from database.models import Prediction
from database.settings import ENGINE
from folium.plugins import TimestampedGeoJson
from geopy.distance import geodesic
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from constants import (
    plants_list,
    plants_dict_geojson,
    lags,
    meteo_params,
    substances,
    Mr,
    threshold
)


# Функция для преобразования формата скорости ветра
def extract_value(value):
    if isinstance(value, (int, float)):
        return value
    match = re.search(r'\((\d+\.\d+)\)', str(value))
    if match:
        return float(match.group(1))
    return value


# Функция загрузки и подготовки данных по нужным веществам из всех 6 геоточек
def load_data(file, all_substances, meteo_params):
    """file - файл с данными для определенной геоточки
       all_substances - названия загрязняющих веществ
    """

    point = pd.read_excel(os.path.join(path, file))
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

    return point


def melt_columns(df, substances, Mr, meteo_params):
    """Функция 'расплавления' столбцов с веществами"""

    df = pd.melt(df, id_vars=meteo_params, ignore_index=False)

    # Закодируем названия веществ их молекулярной массой
    df.replace(substances, Mr, inplace=True)
    df.rename(columns={"variable": "Mr", "value": "C, мг/м³"}, inplace=True)

    df["Mr"] = pd.to_numeric(df["Mr"], errors='coerce')

    return df


def make_df(df, locations_list, num, target_num, lags=None):
    """Функция предобработки датасета с предикторами
       num - номер геоточки
       target_num - номер точки, в которой модель будет считать концентрацию
    """

    coor = locations_list[num - 1]
    target_coor = plants_list[target_num - 1]

    df = df.copy()
    df.dropna(inplace=True)
    df['lat'], df['lon'] = coor

    # Добавим расстояние от данной точки до точки c target_coor в качестве входной переменной
    df['distance, km'] = geodesic(coor, target_coor).km

    # Фичи c временными лагам
    if lags:
        for i in range(1, lags):
            df[f"T внеш_{i}"] = df['T внеш., °C'].shift(i)
            df[f"P атм._{i}"] = df['P атм., мм.рт.ст.'].shift(i)
            df[f"V ветра_{i}"] = df['V ветра, м/с'].shift(i)
            df[f"Угол ветра_{i}"] = df['Угол ветра, °'].shift(i)
            df[f"C, мг/м³_{i}"] = df['C, мг/м³'].shift(i)
    df.dropna(inplace=True)

    return df


def concat_dfs(df_melted_list, locations_list, target_num, lags=5):
    X_list = [make_df(df, locations_list, i + 1, target_num, lags) for i, df in enumerate(df_melted_list)]
    X_num = pd.concat(X_list, axis=0)
    return X_num


def save_to_sqlite(series: pd.Series) -> None:
    """Сохранение данных в базу данных"""

    data_to_db = pd.DataFrame({'time': series.index, 'prediction': series})
    data_to_db['substance'] = substance
    data_to_db['plant'] = plant
    # Преобразование в список словарей
    db_dicts = data_to_db.to_dict(orient='records')
    session.bulk_insert_mappings(Prediction, db_dicts)
    session.commit()


# Границы концентраций для различных веществ
def get_polut_coor(prediction):
    if prediction <= threshold[substance][0]:
        return [0, -90]  # южный полюс
    else:
        return coors  # предприятие


def get_threshold_coor(prediction):
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


if __name__ == "__main__":

    # Пункты контроля воздуха
    locations_list = [
        [55.539306, 051.856451],  # д. Клятле
        [55.622944, 051.825578],  # ул. Ахтубинская, 4б
        [55.650091, 051.852687],  # ул. Гагарина, 32
        [55.598983, 051.771936],  # ул. Юбилейная, 3
        [55.613193, 051.784821],  # ул. Южная, 3
        [55.654578, 051.800072]  # ул. Ямьле, 20
    ]

    # Директория с данными
    path = '../data/raw'
    files = os.listdir(path)

    # Настройки подкючения к sqlite
    engine = create_engine(ENGINE)
    Session = sessionmaker(bind=engine)
    session = Session()

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
        # Свойства объектов GeoJson
        features = []
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
            data['polut_coor'] = data['prediction'].apply(get_polut_coor)
            data['threshold_coor'] = data['prediction'].apply(get_threshold_coor)

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
                        "coordinates": list(data['polut_coor']),
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

        # Сохранение карты для каждого вещества
        map_substance.save(f"templates/map_{substance[:3]}.html")

    session.close()
