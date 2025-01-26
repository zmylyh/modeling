import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy import stats

import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson

try:
    medals = pd.read_csv('data2025/summerOly_medal_counts.csv', encoding='UTF-8')
    hosts = pd.read_csv('data2025/summerOly_hosts.csv', encoding='UTF-8')
    programs = pd.read_csv('data2025/summerOly_programs.csv', encoding='UTF-8')
    athletes = pd.read_csv('data2025/summerOly_athletes.csv', encoding='UTF-8')
except UnicodeDecodeError:
    medals = pd.read_csv('data2025/summerOly_medal_counts.csv', encoding='gbk')
    hosts = pd.read_csv('data2025/summerOly_hosts.csv', encoding='gbk')
    programs = pd.read_csv('data2025/summerOly_programs.csv', encoding='gbk')
    athletes = pd.read_csv('data2025/summerOly_athletes.csv', encoding='gbk')
# 去除BOM及多余字符
hosts.columns = hosts.columns.str.replace('Â', '')
medals.columns = medals.columns.str.replace('Â', '')
programs.columns = programs.columns.str.replace('Â', '')
athletes.columns = athletes.columns.str.replace('Â', '')
hosts.columns = hosts.columns.str.replace('ï»¿', '')
medals.columns = medals.columns.str.replace('ï»¿', '')
programs.columns = programs.columns.str.replace('ï»¿', '')
athletes.columns = athletes.columns.str.replace('ï»¿', '')
hosts = hosts.applymap(lambda x: x.replace('\xa0', '') if isinstance(x, str) else x)
medals = medals.applymap(lambda x: x.replace('\xa0', '') if isinstance(x, str) else x)
programs = programs.applymap(lambda x: x.replace('\xa0', '') if isinstance(x, str) else x)
athletes = athletes.applymap(lambda x: x.replace('\xa0', '') if isinstance(x, str) else x)

print("Medals head:\n", medals.head(), "\n")
print("Hosts head:\n", hosts.head(), "\n")

hosts['Host'] = hosts['Host'].str.replace('Â', '').str.strip()

year_host_dict = {}
for idx, row in hosts.iterrows():
    y = row['Year']
    host_str = row['Host'].split(',')
    if len(host_str) > 1:
        host_country = host_str[-1].strip()
        year_host_dict[y] = host_country
    else:
        year_host_dict[y] = host_str[0].strip()


def check_host(row):
    y = row['Year']
    n = row['NOC']
    if y in year_host_dict:
        if year_host_dict[y].lower() in n.lower():
            return 1
    return 0


medals['IsHost'] = medals.apply(check_host, axis=1)

medals = medals.sort_values(by=['NOC', 'Year']).reset_index(drop=True)

medals['LagTotal'] = medals.groupby('NOC')['Total'].shift(1).fillna(0)
medals['LagTotal2'] = medals.groupby('NOC')['Total'].shift(2).fillna(0)
medals['LagTotal3'] = medals.groupby('NOC')['Total'].shift(3).fillna(0)
medals['LagTotal'] = (medals['LagTotal'] * 0.6 + medals['LagTotal2'] * 0.3 + medals['LagTotal3'] * 0.1)
medals = medals.drop(['LagTotal2', 'LagTotal3'], axis=1)

recent_years = sorted(medals['Year'].unique())[-5:]
recent_medals = medals[medals['Year'].isin(recent_years)]
country_recent_medal_sum = recent_medals.groupby('NOC')['Total'].sum()
never_won_countries = set(country_recent_medal_sum[country_recent_medal_sum == 0].index)

special_cases = {
    'ROC': 'Russia',
    'Soviet Union': 'Russia',
    'Unified Team': 'Russia',
}

for old_noc, new_noc in special_cases.items():
    old_data = medals[medals['NOC'] == old_noc]
    if not old_data.empty:
        medals.loc[medals['NOC'] == old_noc, 'NOC'] = new_noc

medals['NeverWon'] = medals['NOC'].apply(lambda x: 1 if x in never_won_countries else 0)

data_for_model = medals[medals['Year'] >= 1992].copy()
data_for_model['YearFactor'] = data_for_model['Year'] - data_for_model['Year'].min()

X_cols = ['IsHost', 'LagTotal', 'YearFactor']
X = data_for_model[X_cols]
X = sm.add_constant(X)

Z_cols = ['NeverWon']
Z = data_for_model[Z_cols]
Z = sm.add_constant(Z)

y = data_for_model['Total'].values

try:
    zip_model = ZeroInflatedPoisson(endog=y, exog=X, exog_infl=Z)
    zip_res = zip_model.fit(method='nm', maxiter=2000, disp=False)
except:
    try:
        zip_res = zip_model.fit(method='bfgs', maxiter=2000, disp=False)
    except:
        zip_res = zip_model.fit(method='powell', maxiter=2000, disp=False)

print("\nModel Summary:")
print(zip_res.summary())

all_countries = medals['NOC'].unique()
predict_2028 = pd.DataFrame({'NOC': all_countries})

recent_years = sorted(medals['Year'].unique())[-3:]
recent_medals = medals[medals['Year'].isin(recent_years)]
recent_medals_weighted = recent_medals.groupby('NOC').agg({
    'Total': lambda x: np.average(x, weights=[0.6, 0.3, 0.1] if len(x) == 3 else [1] if len(x) == 1 else [0.7, 0.3])
})
year2024_map = dict(zip(recent_medals_weighted.index, recent_medals_weighted['Total']))


def get_lagtotal_avg(noc):
    return year2024_map.get(noc, 0)


predict_2028['IsHost'] = predict_2028['NOC'].apply(lambda x: 1 if 'United States' in x or 'USA' in x else 0)
predict_2028['LagTotal'] = predict_2028['NOC'].apply(get_lagtotal_avg)
predict_2028['NeverWon'] = predict_2028['NOC'].apply(lambda x: 1 if x in never_won_countries else 0)
predict_2028['YearFactor'] = 2028 - data_for_model['Year'].min()

X_2028 = predict_2028[X_cols].values
X_2028 = sm.add_constant(X_2028)

Z_2028 = predict_2028[Z_cols].values
Z_2028 = sm.add_constant(Z_2028)

try:
    predict_2028['PredictedMean'] = zip_res.predict(exog=X_2028, exog_infl=Z_2028, which='mean')
    print("\nPrediction successful!")
except Exception as e:
    print("\nError in prediction:", str(e))


    def predict_medals(row):
        base = row['LagTotal']
        if row['IsHost'] == 1:
            return min(base * 1.3, base + 30)
        elif base > 50:
            return base * 0.95
        elif base > 20:
            return base * 0.98
        elif base > 0:
            return base * 1.05
        elif row['NeverWon'] == 0:
            return 1.0
        else:
            return 0.2


    predict_2028['PredictedMean'] = predict_2028.apply(predict_medals, axis=1)
    print("Using improved simple prediction method!")


def apply_constraints(row):
    base = row['LagTotal']
    if row['IsHost'] == 1:
        return min(row['PredictedMean'], base * 1.3)
    elif base > 0:
        return min(max(row['PredictedMean'], base * 0.7), base * 1.2)
    else:
        return min(row['PredictedMean'], 3)


predict_2028['PredictedMean'] = predict_2028.apply(apply_constraints, axis=1)
predict_2028.to_csv('predict_2028_new.csv', index=False, encoding='utf-8')
print("\n=== Prediction for 2028 (Total Medals) ===")
results = predict_2028[['NOC', 'IsHost', 'LagTotal', 'NeverWon', 'PredictedMean']].sort_values('PredictedMean',
                                                                                               ascending=False)
print("\nTop 30 countries by predicted medals:")
print(results.head(30))

predict_2028['Actual2024'] = predict_2028['NOC'].map(year2024_map).fillna(0)
predict_2028['Diff_2028_2024'] = predict_2028['PredictedMean'] - predict_2028['Actual2024']

improved = predict_2028[predict_2028['Diff_2028_2024'] > 0.5].sort_values('Diff_2028_2024', ascending=False)
declined = predict_2028[predict_2028['Diff_2028_2024'] < -0.5].sort_values('Diff_2028_2024')

print("\n=== Countries likely to improve significantly in 2028 ===")
print(improved[['NOC', 'Actual2024', 'PredictedMean', 'Diff_2028_2024']].head(10))

print("\n=== Countries likely to decline significantly in 2028 ===")
print(declined[['NOC', 'Actual2024', 'PredictedMean', 'Diff_2028_2024']].head(10))

never_won_df = predict_2028[predict_2028['NeverWon'] == 1].copy()
potential_new = never_won_df[never_won_df['PredictedMean'] > 0.2].sort_values('PredictedMean', ascending=False)

print("\n=== Potential new medal countries in 2028 ===")
print(potential_new[['NOC', 'PredictedMean']])

print("\nExpected number of new medalist countries:", len(potential_new))

print("\n=== Analysis Complete ===")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme()

fig = plt.figure(figsize=(20, 16))

ax1 = plt.subplot(111)
top_10_countries = results.head(10)
sns.barplot(data=top_10_countries, x='NOC', y='PredictedMean', palette='viridis', ax=ax1)
ax1.set_title('Top 10 in 2028', fontsize=12)
ax1.set_xlabel('Country')
ax1.set_ylabel('Medals')
ax1.tick_params(axis='x', rotation=45)
plt.show()

ax2 = plt.subplot(111)
host_data = medals[medals['IsHost'] == 1].copy()
host_data['NextTotal'] = host_data.groupby('NOC')['Total'].shift(-1)
host_data['PrevTotal'] = host_data.groupby('NOC')['Total'].shift(1)
host_data = host_data.dropna()

x = np.array([host_data['PrevTotal'], host_data['Total'], host_data['NextTotal']]).T
ax2.boxplot(x, labels=['pre', 'host', 'post'])
ax2.set_title('host impact', fontsize=12)
ax2.set_ylabel('medals')
plt.show()

ax3 = plt.subplot(111)
sns.scatterplot(data=predict_2028, x='LagTotal', y='PredictedMean',
                hue='IsHost', size='NeverWon',
                palette=['#2ecc71', '#e74c3c'], ax=ax3)
ax3.plot([0, predict_2028['LagTotal'].max()],
         [0, predict_2028['LagTotal'].max()],
         'k--', alpha=0.5)
ax3.set_title('predict vs history', fontsize=12)
ax3.set_xlabel('average medals')
ax3.set_ylabel('predicted medals')
plt.show()

ax4 = plt.subplot(111)
sns.histplot(data=predict_2028, x='Diff_2028_2024',
             bins=30, kde=True, color='#3498db', ax=ax4)
ax4.axvline(x=0, color='r', linestyle='--', alpha=0.5)
ax4.set_title('2024-2028 medals changes', fontsize=12)
ax4.set_xlabel('medals change')
ax4.set_ylabel('number of countries')
plt.show()

ax5 = plt.subplot(111)
predict_2028['Level'] = pd.cut(predict_2028['LagTotal'],
                               bins=[0, 5, 20, 50, float('inf')],
                               labels=['weak', 'weak+', 'medium', 'strong'])
sns.boxplot(data=predict_2028, x='Level', y='PredictedMean',
            palette='RdYlBu', ax=ax5)
ax5.set_title('different level predict', fontsize=12)
ax5.set_xlabel('level')
ax5.set_ylabel('predict medals')
plt.show()

ax6 = plt.subplot(111)
sensitivity_factors = np.linspace(0.8, 1.2, 5)
base_predictions = predict_2028['PredictedMean'].copy()
sensitivity_results = []
for factor in sensitivity_factors:
    modified_predictions = base_predictions * factor
    sensitivity_results.append({
        'change rate': f'{(factor - 1) * 100:.0f}%',
        'average': modified_predictions.mean(),
        'max': modified_predictions.max(),
        'median': modified_predictions.median()
    })

sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df.plot(x='change rate', marker='o', ax=ax6)
ax6.set_title('predict and sensitivity', fontsize=12)
ax6.set_xlabel('Parameter change rate')
ax6.set_ylabel('indicator values')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

ax7 = plt.subplot(111)
yearly_total = medals.groupby('Year')['Total'].sum()
sns.regplot(x=yearly_total.index, y=yearly_total.values,
            color='#3498db', ax=ax7)
ax7.set_title('trend of medals', fontsize=12)
ax7.set_xlabel('year')
ax7.set_ylabel('total')
plt.show()

ax8 = plt.subplot(111)
top_countries = results['NOC'].head(6).tolist()
country_data = medals[medals['NOC'].isin(top_countries)]
for country in top_countries:
    country_yearly = country_data[country_data['NOC'] == country]
    plt.plot(country_yearly['Year'], country_yearly['Total'],
             marker='o', label=country)
ax8.set_title('trend', fontsize=12)
ax8.set_xlabel('year')
ax8.set_ylabel('medals')
ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

ax9 = plt.subplot(111)
medals['Level'] = pd.cut(medals['Total'],
                         bins=[0, 5, 20, 50, float('inf')],
                         labels=['weak', 'weak+', 'medium', 'strong'])
level_yearly = medals.pivot_table(index='Year',
                                  columns='Level',
                                  values='Total',
                                  aggfunc='count')
level_yearly_pct = level_yearly.div(level_yearly.sum(axis=1), axis=0)
level_yearly_pct.plot(kind='area', stacked=True,
                      colormap='RdYlBu', ax=ax9)
ax9.set_title('trend of Proportion', fontsize=12)
ax9.set_xlabel('year')
ax9.set_ylabel('Proportion')
plt.show()

ax10 = plt.subplot(111)
n_bootstrap = 1000
bootstrap_predictions = np.zeros((len(predict_2028), n_bootstrap))
for i in range(n_bootstrap):
    noise = np.random.normal(0, 0.1, len(predict_2028))
    bootstrap_predictions[:, i] = predict_2028['PredictedMean'] * (1 + noise)
ci_lower = np.percentile(bootstrap_predictions, 2.5, axis=1)
ci_upper = np.percentile(bootstrap_predictions, 97.5, axis=1)
top_20_idx = predict_2028['PredictedMean'].nlargest(20).index
plt.errorbar(range(20),
             predict_2028.loc[top_20_idx, 'PredictedMean'],
             yerr=[predict_2028.loc[top_20_idx, 'PredictedMean'] - ci_lower[top_20_idx],
                   ci_upper[top_20_idx] - predict_2028.loc[top_20_idx, 'PredictedMean']],
             fmt='o', capsize=5, capthick=1, elinewidth=1)

plt.xticks(range(20), predict_2028.loc[top_20_idx, 'NOC'], rotation=45)
ax10.set_title('top 20 Predicted values and 95 confidence intervals', fontsize=12)
ax10.set_xlabel('country')
ax10.set_ylabel('predict medals')
plt.show()
print("\n=== Code run complete. ===")
