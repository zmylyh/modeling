import pandas as pd

def xlsx_to_csv(path: str, index, output):
    data = pd.read_excel(path, index_col=index)
    data.to_csv(output)
