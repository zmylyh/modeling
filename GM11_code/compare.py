import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

real2024 = pd.read_csv('datasource/summerOly_medal_counts.csv').query('Year==2024')[['NOC', 'Total']]
pre2024 = pd.read_csv('2024full.csv')[['Team', 'Total']]

real2024.columns = ['Team', 'real']
merged = pd.merge(pre2024, real2024, on='Team')
merged['diff'] = merged['Total'] - merged['real']
mean_diff = merged['diff'].mean()


hp = sns.histplot(data=merged['diff'], bins=16)

plt.title('Error distribution of medal prediction for 2024 Paris Olympics')
plt.xlabel('Error interval')
plt.ylabel('Count')

hp.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
hp.axvline(mean_diff, color='red', label=f'mean: {mean_diff:.2f}')

hp.legend()

plt.show()
