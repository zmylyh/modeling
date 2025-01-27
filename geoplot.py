import geopandas as gpd
import geodatasets as dataset
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt

world = gpd.read_file(dataset.get_path('naturalearth.land'))

# get host cities
hosts = pd.read_csv('datasource/summerOly_hosts.csv')
cities = hosts['Host'].tolist()
cities_dict = {}
cities_dict = {cities[i].split(",")[0] : cities[i].split(",")[-1] for i in range(len(cities))}
# for i in range(len(cities)):
#     cities[i] = cities[i].split(",")[0]
years = hosts['Year'].tolist()

# 找出所有城市符合的行
db = pd.read_csv('world_db/worldcities.csv')
db = db[db['city_ascii'].isin(cities_dict.keys())][['city_ascii', 'lat', 'lng', 'country']]
db[['lat', 'lng']] = db[['lat', 'lng']].astype(float) # str -> float
db.columns = ['city', 'latitude', 'longitude', 'country']
for i, row in db.iterrows():
    if row['country'] != cities_dict.get(row['city']):
        db.drop(index=i, inplace=True)
db = db[['city', 'latitude', 'longitude']]

geo = [Point(xy) for xy in zip(db['longitude'], db['latitude'])]
frame = gpd.GeoDataFrame(db, geometry=geo, crs='EPSG:4326')

fig, ax = plt.subplots(1, 1)
world.plot(ax=ax, color='grey')
frame.plot(ax=ax, color='red', markersize=10, label='Host cities')

for x, y, label in zip(frame.geometry.x, frame.geometry.y, frame['city']):
    if label == 'Berlin': 
        x += 4
        y += 5
    if label == 'Helsinki': y += 8
    if label == 'Stockholm':
        x -= 14
        y += 8
    if label == 'London': 
        y += 4
        x -= 12
    if label == 'Amsterdam': 
        y += 7
        x -= 8
    if label == 'Antwerp':
        x += 8
        y += 2
    if label == 'Paris':
        x -= 3
        y += 1
    if label == 'Barcelona':
        x -= 8
    if label == 'St. Louis':
        x -= 15
        y += 3
    ax.text(x + 3, y - 5, label, fontsize=10, ha='center')

plt.legend()
plt.xticks([])
plt.yticks([])
plt.title('Olympic map')
plt.show()
# print(db)
