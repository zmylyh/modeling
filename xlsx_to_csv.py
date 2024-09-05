import pandas as pd

def xlsx_to_csv(path: str, index_col, output):
    data = pd.read_excel(path, index_col)
    data.to_csv(output, enconding='utf-8')
