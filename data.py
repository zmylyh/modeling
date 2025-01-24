import pandas as pd

def getMedal():
    raw = pd.read_csv('data2025/summerOly_athletes.csv')
    data = raw[['Team', 'Sport', 'Event', 'Year', 'Medal']]
    data.sort_values(by=['Team', 'Sport', 'Event', 'Year', 'Medal'])
    data.to_csv('after.csv', index=False)

getMedal()
