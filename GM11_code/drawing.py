import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('2028full.csv')
data_teamSport = pd.read_csv('2028teamfull.csv')

# plot_data = data.head(10)
# plot_data = data_teamSport.query('Team=="United States"').sort_values(by='Total', ascending=False).head(10)
# sns.barplot(data=plot_data, x='Team', y='Total', palette='deep')
# bp = sns.barplot(data=plot_data, x='Sport', y='Total', palette='deep')

# for i, row in plot_data.iterrows():
    # plt.text(i, row['Total'] + 0.5, str(int(row['Total'])), ha='center', va='bottom')

# for p in bp.patches:
#     bp.text(
#         p.get_x() + p.get_width() / 2,
#         p.get_height() + 0.1,
#         str(int(p.get_height())),
#         ha='center',
#         va='bottom'
#     )

# plt.title('Predicted top 10 teams of 2028 Los Angeles Olympics')
# plt.title('Predicted 10 top sports of Team United States in 2028 Olympics')
# plt.ylabel('Total medals')

topTeam = data.head(10)['Team'].tolist()
medal = pd.read_csv('datasource/summerOly_medal_counts.csv')
medal = medal[medal['NOC'].isin(topTeam)]

print(medal)

sns.lineplot(data=medal, x='Year', y='Total', hue='NOC', marker='o')
plt.title('Top 10 teams medal counts for all Olympics')
plt.ylabel('Total medals')
plt.legend(title='Team')
plt.show()