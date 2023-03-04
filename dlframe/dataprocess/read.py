import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import dlframe.infrastructure.pytorch_utils as ptu
import dlframe.dataprocess.preprocess as datapre

def read_csv(path, mode):
    if mode=='test':
        data = pd.read_csv(f'{path}/covid.test.csv').drop(columns='id').values
        data = datapre.zscore_norm(data)
        return data

    df = pd.read_csv(f'{path}/covid.train.csv').drop(columns='id')
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=ptu.seed)
    if mode=='eval':
        data = eval_df.iloc[:, :-1].values
        target = eval_df.iloc[:, -1].values
    elif mode=='train':
        data = train_df.iloc[:, :-1].values
        target = train_df.iloc[:, -1].values
    data = datapre.zscore_norm(data)
    target = target.reshape(-1, 1)
    return data, target

def read_parquet(path, start_date, end_date, feature, y_name, n_jobs=32):
    """
    path: The path to the folder which contains parquet files.
    """
    files = os.listdir(path)
    files = [f'{path}/{file}' for file in files if file.split('.')[0] >= start_date and file.split('.')[0] <= end_date]
    columns = ['ticker', 'date'] + feature + y_name
    with Pool(n_jobs) as p:
        temp = p.map(pd.read_parquet, files)
    df = (
        pd.concat(temp, axis=0)
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0, subset=[y_name])
    )
    data = df.values
    return data