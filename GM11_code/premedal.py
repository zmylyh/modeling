import pandas as pd

data = pd.read_csv('2024cleaned.csv')
sport_table = pd.read_csv('sportcheck.csv')


def calc(row):
    sport = row['Sport']
    count = sport_table.query('Sport==@sport')['count'].tolist()[0]
    if float(count) < 0: count = 0
    row['pgold'] = float(row['goldPre']) * float(count)
    row['psilver'] = float(row['silverPre']) * float(count)
    row['pbronze'] = float(row['bronzePre']) * float(count)
    try: row['pgold'] = int(row['pgold'])
    except: pass
    try: row['psilver'] = int(row['psilver'])
    except: pass
    try: row['pbronze'] = int(row['pbronze'])
    except: pass
    return row

data['pgold'] = ''
data['psilver'] = ''
data['pbronze'] = ''

data = data.apply(calc, axis=1)
# data.groupby('Team')['pgold', 'psilver', 'pbronze'].sum().reset_index()
real = data.groupby('Team')[['pgold', 'psilver', 'pbronze']].sum().reset_index()
real_sorted = real.sort_values(by=['pgold', 'psilver', 'pbronze'], ascending=False)
real_sorted['Total'] = real_sorted['pgold'] + real_sorted['psilver'] + real_sorted['pbronze']
real_sorted = real_sorted.sort_values(by=['Total'], ascending=False)
# print(real_sorted)
real_sorted.to_csv('2024full.csv', index=False)

byteam = data.groupby(['Team', 'Sport'])[['pgold', 'psilver', 'pbronze']].sum().reset_index()
byteam['Total'] = byteam['pgold'] + byteam['psilver'] + byteam['pbronze']

byteam.to_csv('2024teamfull.csv', index=False)