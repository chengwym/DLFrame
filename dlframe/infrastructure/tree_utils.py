import xgboost as xgb
import lightgbm as lgb

seed = 20031122

def set_seed(seed=20031122):
    pass

def xgb_dmatrix(data, label=None):
    return xgb.DMatrix(data=data, label=label)
