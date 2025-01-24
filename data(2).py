import pandas as pd

def getMedal():
    raw = pd.read_csv('data2025/summerOly_athletes.csv')
    data_cols = ['Team', 'Sport', 'Year', 'Medal']
    data = raw[data_cols]
    data_sorted = data.sort_values(by=data_cols)

    new = data_sorted.groupby(['Team', 'Year', 'Sport', 'Medal']).size().unstack(fill_value=0).reset_index()
    new['Total'] = new['Gold'] + new['Silver'] + new['Bronze'] + new['No medal']
    output = new[['Team', 'Year', 'Sport', 'Gold', 'Silver', 'Bronze', 'Total']]
    output.to_csv('newafter.csv', index=False)

getMedal()
