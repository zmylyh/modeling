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

medals = medals.sort_values(by=['NOC','Year']).reset_index(drop=True)
medals['LagTotal'] = 0
grouped = medals.groupby('NOC')

medals['LagTotal'] = grouped['Total'].shift(1).fillna(0)
country_medal_sum = medals.groupby('NOC')['Total'].sum()
never_won_countries = set(country_medal_sum[country_medal_sum == 0].index)
medals['NeverWon'] = medals['NOC'].apply(lambda x: 1 if x in never_won_countries else 0)
data_for_model = medals.copy()

data_for_model['YearFactor'] = data_for_model['Year']-data_for_model['Year'].min()

X_cols = ['IsHost', 'LagTotal', 'YearFactor']

X = data_for_model[X_cols]
X = sm.add_constant(X)

Z_cols = ['NeverWon']
Z = data_for_model[Z_cols]
Z = sm.add_constant(Z, has_constant='add')

y = data_for_model['Total'].values

zip_model = ZeroInflatedPoisson(endog=y, exog=X, exog_infl=Z, p=0)
zip_res = zip_model.fit(method='bfgs', maxiter=200, disp=False)
print(zip_res.summary())

predicted_mean = zip_res.predict(X, exog_infl=Z, which='mean')
residuals = y-predicted_mean

evaluation_df = data_for_model[['NOC','Year','Total']].copy()
evaluation_df['Predicted'] = predicted_mean
print(evaluation_df.head(20))

all_countries = medals['NOC'].unique()
predict_2028 = pd.DataFrame({'NOC': all_countries})
year2024 = medals[medals['Year'] == 2024][['NOC','Total']]
year2024_map = dict(zip(year2024['NOC'], year2024['Total']))
def get_lagtotal_2024(noc):
    return year2024_map[noc] if noc in year2024_map else 0
def check_host_2028(noc):

    if 'United States' in noc or 'USA' in noc:
        return 1

    return 0
def check_neverwon(noc):
    return 1 if noc in never_won_countries else 0

predict_2028['IsHost'] = predict_2028['NOC'].apply(check_host_2028)
predict_2028['LagTotal'] = predict_2028['NOC'].apply(get_lagtotal_2024)
predict_2028['NeverWon'] = predict_2028['NOC'].apply(check_neverwon)

year_factor_2028 = 2028-data_for_model['Year'].min()
predict_2028['YearFactor'] = year_factor_2028

X_2028 = predict_2028[['IsHost','LagTotal','YearFactor']]
X_2028 = sm.add_constant(X_2028, has_constant='add')

Z_2028 = predict_2028[['NeverWon']]
Z_2028 = sm.add_constant(Z_2028,has_constant='add')


predict_2028['PredictedMean'] = zip_res.predict(exog=X_2028, exog_infl=Z_2028, which='mean')

print("\n=== Prediction for 2028 (Total Medals) ===")
print(predict_2028[['NOC','IsHost','LagTotal','NeverWon','PredictedMean']].sort_values('PredictedMean', ascending=False))
sorted_predict_2028 = predict_2028[['NOC','PredictedMean']].sort_values('PredictedMean', ascending=False)
#sorted_predict_2028.to_csv('sorted_predict_2028.csv', index=False, encoding='utf-8')
predict_2028['Actual2024'] = predict_2028['NOC'].map(year2024_map).fillna(0)
predict_2028['Diff_2028_2024'] = predict_2028['PredictedMean']-predict_2028['Actual2024']
improved = predict_2028[predict_2028['Diff_2028_2024']>0]
declined = predict_2028[predict_2028['Diff_2028_2024']<0]
print("\nCountries likely to improve in 2028:", improved['NOC'].values)
print("Countries likely to decline in 2028:", declined['NOC'].values)
main_params = zip_res.params[:-len(Z_cols)-1] 
infl_params = zip_res.params[-len(Z_cols)-1:]

threshold = 0.5 
predict_2028['Prob_Get_Medal'] = np.nan 

def logistic(x):
    return 1.0/(1.0 + np.exp(-x))

poisson_coef = zip_res.params[:len(X_cols)+1] 
infl_coef = zip_res.params[len(X_cols)+1:] 

Xmat_2028 = X_2028.values
Zmat_2028 = Z_2028.values

log_lambda_2028 = Xmat_2028 @ poisson_coef
lambda_2028 = np.exp(log_lambda_2028)

infl_2028 = Zmat_2028 @ infl_coef
pi_2028 = logistic(infl_2028)

p_zero_poisson = np.exp(-lambda_2028)

p_zero_ZIP = pi_2028 + (1-pi_2028) * p_zero_poisson

predict_2028['Prob_Get_Medal'] = 1-p_zero_ZIP
never_won_df = predict_2028[predict_2028['NeverWon']==1].copy()
print("\n=== Potential new medal countries in 2028 ===")
print(never_won_df[['NOC','Prob_Get_Medal','PredictedMean']].sort_values('Prob_Get_Medal', ascending=False))
mean_new_medal = never_won_df['Prob_Get_Medal'].sum()
print(f"\nExpected number of new medalist countries (approx) in 2028: {mean_new_medal:.2f}")

print("\n=== Code run complete. ===")













