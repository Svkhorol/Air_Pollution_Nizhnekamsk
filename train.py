import copy
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from geopy.distance import geodesic


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
    indexes = list(point.loc[0])
    indexes = indexes[1:]
    point = point.loc[3:].set_index('Интервал отбора')
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


def make_df(df, num, target_num, lags=None):
    """Функция предобработки датасета с предикторами
       num - номер геоточки
       target_num - номер точки, в которой модель будет считать концентрацию
    """

    coor = locations_list[num - 1]
    target_coor = locations_list[target_num - 1]

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


def concat_dfs(df_melted_list, num, lags=5):
    X_list = [make_df(df, i + 1, num, lags) for i, df in enumerate(df_melted_list)]
    del X_list[num - 1]  # сопоставление точки самой с собой
    X_num = pd.concat(X_list, axis=0)
    return X_num


SEED = 10
path = 'data/raw'
files = os.listdir(path)

locations_list = [
    [55.539306, 051.856451],    # д. Клятле
    [55.622944, 051.825578],    # ул. Ахтубинская, 4б
    [55.650091, 051.852687],    # ул. Гагарина, 32
    [55.598983, 051.771936],    # ул. Юбилейная, 3
    [55.613193, 051.784821],    # ул. Южная, 3
    [55.654578, 051.800072]     # ул. Ямьле, 20
]
meteo_params = ['T внеш., °C',
                'P атм., мм.рт.ст.',
                'V ветра, м/с',
                'Угол ветра, °',
                'Направление ветра'
                ]
# Загрязнители в алфавитном порядке
substances = ['CO, мг/м³',
              'H2S, мг/м³',
              'NH3, мг/м³',
              'NO, мг/м³',
              'NO2, мг/м³',
              'SO2, мг/м³'
              ]
# Молекулярные массы веществ из substances
Mr = [28, 30, 46, 17, 64, 34]

# Временной лаг
lags = 4

# Загрузка гиперпараметров
with open('hyperparams.json', 'r') as f:
    hyperparams_dict = json.load(f)

# Загрузка, обработка и объединение данных
points = [load_data(file, substances, meteo_params) for file in files if file.endswith('.xlsx')]
df_melted_list = [melt_columns(df, substances, Mr, meteo_params) for df in points]
X_num_list = [concat_dfs(df_melted_list, j + 1, lags) for j in range(len(df_melted_list))]

# Обучение Catboost для каждого из веществ
for substance in substances:

    data_with_substance = []
    X_list_substance = copy.deepcopy(X_num_list)

    # Добавление целевой переменной
    for (num, X_num), point in zip(enumerate(X_list_substance), points):
        if substance in list(point):
            data_with_substance.append(X_num)
            data_with_substance[-1][f'target_{substance}'] = point[substance]

    X = pd.concat(data_with_substance, axis=0)
    X.dropna(inplace=True)
    X.drop_duplicates(inplace=True)
    X = pd.get_dummies(X, columns=['Направление ветра'])
    y = X.pop(f'target_{substance}')

    catboost_params = hyperparams_dict.get(substance, {}).get('best_params', {})
    catboost = CatBoostRegressor(**catboost_params, random_state=SEED)
    catboost.fit(X, y, early_stopping_rounds=200, verbose=0)

    # Сохранение обученной модели
    with open(f'app/ml_models/model_{substance[:3]}.pkl', 'wb') as file:
        pickle.dump(catboost, file)
