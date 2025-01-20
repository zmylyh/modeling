import pandas as pd

# 读取 CSV 文件
csv_file = 'csv/result2.csv'
df = pd.read_csv(csv_file)

# 将 DataFrame 保存为 Excel 文件
excel_file = 'csv/result2.xlsx'
df.to_excel(excel_file, index=False)

print(f"CSV 文件已成功转换为 Excel 文件：{excel_file}")