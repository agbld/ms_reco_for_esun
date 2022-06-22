#%%
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
tqdm.pandas()

#%%
# list all csv files in the directory
files = os.listdir('./')
# filter out the csv files
files = [f for f in files if f.endswith('.csv')]

df = pd.read_csv(files[0])
for file in files:
    tmp_df = pd.read_csv(file)
    tmp_df.replace(to_replace=-1, value=np.NaN, inplace=True)
    tmp_df['precision_at_5'] = round(tmp_df['precision_at_5'] * 100, 2)
    tmp_df.rename(columns={'precision_at_5': file[:-4]}, inplace=True)
    df = df.join(tmp_df.set_index('cust_no'), on='cust_no')
df.drop('precision_at_5', axis=1, inplace=True)

#%%
# filter columns by cond_str
cond_str = '2019-01-31_18'
# cond_str = '__1024__'
# cond_str = ''
drop_columns = []
for col in df.columns:
    if not cond_str in col:
        drop_columns.append(col)
filter_df = df.drop(drop_columns, axis=1)

#%%
# study
filter_df.dropna(axis=0, how='all', inplace=True)
filter_df['sum'] = filter_df.sum(axis=1)
filter_df = filter_df[filter_df['sum'] > 0]
filter_df.drop('sum', axis=1, inplace=True)

def mae(x: pd.Series) -> float:
    y = x[1:].dropna()
    y = y - y.mean()
    y = np.abs(y)
    return y.mean()
filter_df['mean_abs_error'] = filter_df.progress_apply(mae, axis=1)

#%%