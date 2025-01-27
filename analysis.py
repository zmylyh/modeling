import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_theme()

try:
    font = FontProperties(fname=r"C:\Windows\Fonts\SimHei.ttf")
except:
    print("未找到中文字体，使用系统默认字体")
    font = FontProperties()
try:
    medals = pd.read_csv('data2025/summerOly_medal_counts.csv', encoding='utf-8')
    hosts = pd.read_csv('data2025/summerOly_hosts.csv', encoding='utf-8')
    programs = pd.read_csv('data2025/summerOly_programs.csv', encoding='utf-8')
    athletes = pd.read_csv('data2025/summerOly_athletes.csv', encoding='utf-8')
except UnicodeDecodeError:
    medals = pd.read_csv('data2025/summerOly_medal_counts.csv', encoding='gbk')
    hosts = pd.read_csv('data2025/summerOly_hosts.csv', encoding='gbk')
    programs = pd.read_csv('data2025/summerOly_programs.csv', encoding='gbk')
    athletes = pd.read_csv('data2025/summerOly_athletes.csv', encoding='gbk')


hosts.columns = hosts.columns.str.replace('ï»¿', '')
medals.columns = medals.columns.str.replace('ï»¿', '')
programs.columns = programs.columns.str.replace('ï»¿', '')
athletes.columns = athletes.columns.str.replace('ï»¿', '')

print("=== Data Structure Info ===")
print("\nMedals columns:", medals.columns.tolist())
print("\nHosts columns:", hosts.columns.tolist())
print("\nPrograms columns:", programs.columns.tolist())
print("\nAthletes columns:", athletes.columns.tolist())

def analyze_medals_advanced():
    plt.figure(figsize=(15, 12))
 
    plt.subplot(1,1,1)
    top_countries = medals.groupby('NOC')['Total'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_countries.index, y=top_countries.values)
    plt.title('Top 10 Countries by Total Medals')
    plt.xticks(rotation=45)
    plt.show()
    

    plt.subplot(1,1,1)
    recent_medals = medals[medals['Year'] >= 2000]
    medal_ratios = recent_medals.groupby('Year')[['Gold', 'Silver', 'Bronze']].sum()
    medal_ratios_pct = medal_ratios.div(medal_ratios.sum(axis=1), axis=0)
    medal_ratios_pct.plot(kind='area', stacked=True)
    plt.title('Medal Distribution Trend (2000-Present)')
    plt.grid(True)
    plt.show()
    plt.subplot(1,1,1)
    medal_rates = medals.groupby('NOC')['Total'].sum() / len(medals['Year'].unique())
    sns.histplot(medal_rates, kde=True)
    plt.title('Distribution of Medal Winning Rates')
    plt.show()
    plt.subplot(1,1,1)
    host_advantage = []
    
    print("\nHosts data preview:")
    print(hosts.head())
    
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
                labels=['Host Countries', 'Non-Host Countries'])
    plt.title('Host Country Advantage Analysis')
    
    plt.show()

def analyze_sports_advanced():
    plt.figure(figsize=(15, 12))
   
    plt.subplot(1,1,1)
    sport_diversity = athletes.groupby('Year')['Sport'].nunique()
    sns.regplot(x=sport_diversity.index, y=sport_diversity.values)
    plt.title('Sport Diversity Trend Over Time')
    plt.show()

    plt.subplot(1,1,1)
    gender_sport_pivot = pd.crosstab(athletes['Sport'], athletes['Sex'])
    sns.heatmap(gender_sport_pivot, cmap='YlOrRd', annot=True, fmt='d')
    plt.title('Gender Participation Heatmap by Sport')
    plt.show()

    plt.subplot(1,1,1)
    medal_rates = athletes[athletes['Medal'].notna()].groupby('Sport')['Medal'].count() / \
                 athletes.groupby('Sport').size()
    medal_rates.sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title('Top 10 Sports by Medal Success Rate')
    plt.xticks(rotation=45)
    plt.show()
    
    plt.subplot(1,1,1)
    sport_features = pd.get_dummies(athletes[['Sport', 'Sex', 'Medal']])
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    sport_pca = pca.fit_transform(scaler.fit_transform(sport_features))
    plt.scatter(sport_pca[:, 0], sport_pca[:, 1], alpha=0.5)
    plt.title('Sport Clustering Analysis (PCA)')
    plt.show()
    

def analyze_hosts_advanced():
    plt.figure(figsize=(15, 10))

    hosts['Host'] = hosts['Host'].str.replace('Â', '').str.strip()
    host_counts = hosts['Host'].str.split(',').str[-1].str.strip().value_counts()
    
    plt.subplot(1, 1, 1)
    plt.pie(host_counts.values, labels=host_counts.index, autopct='%1.1f%%')
    plt.title('Olympic Host Countries Distribution')
    plt.show()
    
    plt.subplot(1,1,1)
    host_years = hosts['Year'].astype(int).values
    host_total_medals = []
    for year in host_years:
        total = medals[medals['Year'] == year]['Total'].sum()
        host_total_medals.append(total)
    
    plt.plot(host_years, host_total_medals, marker='o')
    plt.title('Total Medals Trend in Olympic Years')
    plt.xlabel('Year')
    plt.ylabel('Total Medals')
    plt.show()
    
    


def analyze_athletes_advanced():
    plt.figure(figsize=(15, 12))
    
    
    plt.subplot(1,1,1)
    medalists = athletes[athletes['Medal'].notna()]
    non_medalists = athletes[athletes['Medal'].isna()]
    years = sorted(athletes['Year'].unique())
    medalist_counts = [len(medalists[medalists['Year'] == year]) for year in years]
    non_medalist_counts = [len(non_medalists[non_medalists['Year'] == year]) for year in years]
    
    plt.plot(years, medalist_counts, label='Medalists', marker='o')
    plt.plot(years, non_medalist_counts, label='Non-medalists', marker='o')
    plt.title('Athlete Participation Trend: Medalists vs Non-medalists')
    plt.xlabel('Year')
    plt.ylabel('Number of Athletes')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
    plt.subplot(1,1,1)
    gender_ratio = pd.crosstab(athletes['Year'], athletes['Sex'])
    gender_ratio_pct = gender_ratio.div(gender_ratio.sum(axis=1), axis=0)
    gender_ratio_pct.plot(kind='area', stacked=True)
    plt.title('Gender Ratio Evolution Over Time')
    plt.show()
  
    plt.subplot(1,1,1)
    gold_medalists = athletes[athletes['Medal'] == 'Gold']
    top_athletes = gold_medalists['Name'].value_counts().head(10)
    sns.barplot(x=top_athletes.values, y=top_athletes.index)
    plt.title('Top 10 Athletes by Gold Medals')
    plt.show()
 
    plt.subplot(1,1,1)
    noc_diversity = athletes.groupby('Year')['NOC'].nunique()
    sns.regplot(x=noc_diversity.index, y=noc_diversity.values)
    plt.title('Country Participation Diversity Trend')
    plt.show()
    
   

if __name__ == "__main__":
    analyze_medals_advanced()
    analyze_sports_advanced()
    analyze_hosts_advanced()
    analyze_athletes_advanced()

    print("=== Olympic Games Analysis Report ===")
    print("\nTotal Olympics analyzed:", len(hosts))
    print("Total athletes analyzed:", len(athletes))
    print("Number of sports:", len(athletes['Sport'].unique()))
    print("Participating countries:", len(athletes['NOC'].unique()))
   
    print("\nAdvanced Statistical Analysis:")
    print("Medal distribution normality test:", 
          stats.normaltest(medals['Total'].values))
    print("Correlation between Gold and Total medals:", 
          stats.pearsonr(medals['Gold'], medals['Total'])) 