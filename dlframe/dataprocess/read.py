import pandas as pd
from sklearn.model_selection import train_test_split

import dlframe.infrastructure.pytorch_utils as ptu

def read_csv(path, mode):
    if mode=='test':
        data = ptu.from_numpy(pd.read_csv(f'{path}/covid.test.csv').drop(columns='id').values)
        data = (data-data.mean())/data.std()
        return data

    df = pd.read_csv(f'{path}/covid.train.csv').drop(columns='id')
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=ptu.seed)
    if mode=='eval':
        data = ptu.from_numpy(eval_df.iloc[:, :-1].values)
        target = ptu.from_numpy(eval_df.iloc[:, -1].values)
    elif mode=='train':
        data = ptu.from_numpy(train_df.iloc[:, :-1].values)
        target = ptu.from_numpy(train_df.iloc[:, -1].values)
    data = (data-data.mean())/data.std()
    return data, target