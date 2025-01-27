import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()

file_path = 'medal_data.csv'
medals = pd.read_csv(file_path, encoding='UTF-8')

australia_swimming = medals[(medals['Team'] == 'Australia') & (medals['Sport'] == 'Swimming')]

start_year = 1960
end_year = 1980
australia_swimming = australia_swimming[(australia_swimming['Year'] >= start_year) & (australia_swimming['Year'] <= end_year)]

australia_swimming['Tota'] = australia_swimming['gs'] + australia_swimming['ss'] + australia_swimming['bs']

plt.figure(figsize=(10, 6))
plt.plot(australia_swimming['Year'], australia_swimming['Tota'], marker='o', linestyle='-', color='b')
plt.title('Total Medals in Swimming for Australia (1960-2000)')
plt.xlabel('Year')
plt.ylabel('Total Medals')
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data2025/summerOly_athletes.csv'
athletes = pd.read_csv(file_path, encoding='UTF-8')

filtered_data = athletes[(athletes['Year'] >= 1960) & (athletes['Year'] <= 1980) &
                         (athletes['Team'] == 'Australia') & (athletes['Sport'] == 'Swimming')]


yearly_counts = filtered_data.groupby('Year').size()


plt.figure(figsize=(10, 6))
plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-', color='b')
plt.title('Number of Australian Swimmers (1960-1980)')
plt.xlabel('Year')
plt.ylabel('Number of Athletes')
plt.grid(True)
plt.show()