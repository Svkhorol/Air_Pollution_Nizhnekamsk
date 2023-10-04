import folium


# Список мест отбора проб
locations = {
    'д. Клятле': [55.539306, 051.856451],
    'ул. Ахтубинская, 4б': [55.622944, 051.825578],
    'ул. Гагарина, 32': [55.650091, 051.852687],
    'ул. Юбилейная, 3': [55.598983, 051.771936],
    'ул. Южная, 3': [55.613193, 051.784821],
    'ул. Ямьле, 20': [55.654578, 051.800072]
}

# Создание и сохранение главной карты
main_map = folium.Map(location=[55.613193, 051.784821], zoom_start=11.5)

# Добавление маркеров
for marker, coordinates in locations.items():
    folium.Marker(location=coordinates,
                  popup=marker,
                  icon=folium.Icon(color='green')
                  ).add_to(main_map)

main_map.save("../app/templates/main_map.html")
