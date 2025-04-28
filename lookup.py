import pandas as pd


def filter_rows_by_column_for_str_features(df, column_name, value):
    result = df[df[column_name] == value]
    return result if not result.empty else None

def filter_rows_by_column_for_num_features(df, column_name, value, fuzzy_precision=None, fuzzy_size=None, avoid_infinite_loop=1000):
    result = df[df[column_name] == value]
    if fuzzy_size and fuzzy_precision:
        count=0
        while len(result)<fuzzy_size:
            count+=1
            result = df[abs(df[column_name]-value) <=count*fuzzy_precision]
            if count>=avoid_infinite_loop or len(result)>=len(df):
                break
    return result if not result.empty else None