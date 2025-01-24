import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_theme()

medals = pd.read_csv('data2025/summerOly_medal_counts.csv', encoding='latin1')
hosts = pd.read_csv('data2025/summerOly_hosts.csv', encoding='latin1')
programs = pd.read_csv('data2025/summerOly_programs.csv', encoding='latin1')
athletes = pd.read_csv('data2025/summerOly_athletes.csv', encoding='latin1')
hosts.columns = hosts.columns.str.replace('ï»¿', '')
medals.columns = medals.columns.str.replace('ï»¿', '')
programs.columns = programs.columns.str.replace('ï»¿', '')
athletes.columns = athletes.columns.str.replace('ï»¿', '')

# 加起来最多的10个国家
plt.figure()
top_countries = medals.groupby('NOC')['Total'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_countries.index, y=top_countries.values)
plt.title('Top 10 Countries by Total Medals')
plt.xticks(rotation=45)
plt.show()

# 金牌的占比
recent_medals = medals[medals['Year'] >= 2000]
medal_ratios = recent_medals.groupby('Year')[['Gold', 'Silver', 'Bronze']].sum()
medal_ratios_pct = medal_ratios.div(medal_ratios.sum(axis=1), axis=0)
medal_ratios_pct.plot(kind='area', stacked=True)
plt.title('Medal Distribution Trend (2000-Present)')
plt.grid(True)
plt.show()

# 获得率
medal_rate = medals.groupby('NOC')['Total'].sum() / len(medals['Year'].unique())
sns.histplot(medal_rate, kde=True)
plt.title('Distribution of Medal Winning Rates')
plt.show()

# 主办国nb
host_advantage = []
for _, row in hosts.iterrows():
    host_cols = row.index.tolist()
    if 'year' in [col.lower() for col in host_cols]:
        year_col = [col for col in host_cols if col.lower() == 'year'][0]
        host_col = [col for col in host_cols if col.lower() == 'host'][0]
        
        year = row[year_col]
        host_country = row[host_col].split(',')[-1].strip()
        host_medals = medals[(medals['Year'] == year) & 
                            (medals['NOC'].str.contains(host_country))]['Total'].values
        if len(host_medals) > 0:
            host_advantage.append(host_medals[0])

plt.boxplot([host_advantage, 
            medals[~medals['NOC'].isin(hosts['Host'])]['Total'].values],
            labels=['Host Countries',''])
plt.title('Host Country Advantage Analysis')
plt.show()




