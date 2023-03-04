import os
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

def xgb_train_eval(params, train_dmatrix, eval_dmatrix, log_dir, plot=True, savemodel=False):
    result = {}
    model = xgb.train(
        params=params,
        dtrain=train_dmatrix,
        num_boost_round=params['num_boost_round'],
        evals=[(eval_dmatrix, 'eval'), (train_dmatrix, 'train')],
        evals_result=result
    )
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    if plot:
        plt.figure()
        plt.title('loss curve')
        plt.plot(result['train'][params['eval_metric']], 'r')
        plt.plot(result['eval'][params['eval_metric']], 'b')
        plt.legend(['train loss', 'eval loss'])
        plt.savefig(f'{log_dir}/loss.png')
        plt.close()
        
    if savemodel:
        model.save_model(f'{log_dir}/model.pt')
        
def xgb_train(params, train_dmatrix, test_dmatrix, log_dir, plot=True, savemodel=False):
    result = {}
    model = xgb.train(
        params=params,
        dtrain=train_dmatrix,
        num_boost_round=params['num_boost_round'],
        evals=[(train_dmatrix, 'train')],
        evals_result=result
    )
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        
    y_predict = model.predict(test_dmatrix)
    
    if plot:
        plt.figure()
        plt.title('loss curve')
        plt.plot(result['train'][params['eval_metric']], 'r')
        plt.legend(['train loss'])
        plt.savefig(f'{log_dir}/loss.png')
        plt.close()
        
    if savemodel:
        model.save_model(f'{log_dir}/model.pt')
        
    return y_predict
        
def eval(model, test_dmatrix):
    pass