import pandas as pd

# 函数：清理数字，使其在 0 和 1 之间
def cleanNum(n):
    if float(n) > 1.0:
        return '1.0'
    if float(n) < 0.0:
        return '0.0'
    return n

# 读取数据
pre_data = pd.read_csv('2024pre.csv')

# 直接修改列数据
pre_data['goldPre'] = pre_data['goldPre'].apply(cleanNum)
pre_data['silverPre'] = pre_data['silverPre'].apply(cleanNum)
pre_data['bronzePre'] = pre_data['bronzePre'].apply(cleanNum)

# 保存清理后的数据
pre_data.to_csv('2024cleaned.csv', index=False)

# pre_data['Sport'].drop_duplicates().to_csv('2028sport.csv', index=False)

# 修复sport和program里面的sport不一定对应问题
# sport_data = pd.read_csv('2028sport.csv')
# programs = pd.read_csv('./datasource/summerOly_programs.csv')
# sport_total = programs.groupby(['Sport', 'Discipline'])['2024'].sum().reset_index()#.drop([42, 43, 44])
# sport_total.to_csv('sport_total.csv', index=False)
