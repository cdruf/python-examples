import pandas as pd

from util.data_helper import to_excel_from_dfs

if __name__ == "__main__":
    df1 = pd.DataFrame({'a': [1, 2, 3],
                        'b': [4, 5, 6]})
    df2 = pd.DataFrame({'a': [7, 8, 9],
                        'b': [4, 5, 6]})
    to_excel_from_dfs("./test.xlsx", dfs=[df1, df2], sheet_names=['a', 'b'])
