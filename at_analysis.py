import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# 读取奖牌数据
medal_file_path = 'medal_data.csv'
medals = pd.read_csv(medal_file_path, encoding='UTF-8')

# 过滤出澳大利亚在游泳项目的数据
australia_swimming_medals = medals[(medals['Team'] == 'Australia') & (medals['Sport'] == 'Swimming')]

# 限制时间范围
start_year = 1900
end_year = 2024
australia_swimming_medals = australia_swimming_medals[(australia_swimming_medals['Year'] >= start_year) & (australia_swimming_medals['Year'] <= end_year)]

# 计算奖牌总数
australia_swimming_medals['Total'] = australia_swimming_medals['Gold'] + australia_swimming_medals['Silver'] + australia_swimming_medals['Bronze']

# 读取运动员数据
athlete_file_path = 'data2025/summerOly_athletes.csv'
athletes = pd.read_csv(athlete_file_path, encoding='UTF-8')
p = pd.read_csv('medal_data.csv', encoding='utf-8')

# 过滤出澳大利亚在游泳项目的数据
filtered_data = athletes[(athletes['Year'] >= 1900) & (athletes['Year'] <= 2024) &
                         (athletes['Team'] == 'Australia') & (athletes['Sport'] == 'Swimming')]
filtered_data1 = filtered_data
# 根据Name列去重，确保每个运动员只统计一次
filtered_data = filtered_data.drop_duplicates(subset='Name')

# 统计每年参与运动员人数
yearly_counts = filtered_data.groupby('Year').size()
y = filtered_data1.groupby('Year').size()
#gc = p[(p['Team'] == 'Australia') & (p['Sport'] == 'Swimming') & (p['Year'] >= 1900) & (p['Year'] <= 2024)]
# 创建图形
plt.figure(figsize=(10, 6))

# 绘制奖牌总数折线图
plt.plot(australia_swimming_medals['Year'], australia_swimming_medals['Total'], marker='o', linestyle='-', color='b', label='Total Medals')

# 绘制参与运动员人数折线图
plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-', color='r', label='Number of Athletes')
plt.plot(yearly_counts.index, y, marker='o', linestyle='-', color='y', label='Number of program')

# 设置标题和标签
plt.title('Total Medals and Number of Athletes in Swimming for Australia (1900-2024)')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()