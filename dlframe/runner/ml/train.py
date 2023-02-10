import argparse

import dlframe.infrastructure.tree_utils as tru
import dlframe.dataprocess.read as rd
from dlframe.runner.ml.method import xgb_train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--min_child_weight', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--lambda', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--num_boost_round', type=int, default=100)
    parser.add_argument('--eval_metric', type=str, default='rmse')
    
    args = parser.parse_args()
    params = vars(args)
    
    feature_train, target_train = rd.read_csv('./', 'train')
    feature_eval, target_eval = rd.read_csv('./', 'eval')
    feature_test = rd.read_csv('./', 'test')
    
    train_dmatrix = tru.xgb_dmatrix(feature_train, target_train)
    eval_dmatrix = tru.xgb_dmatrix(feature_eval, target_eval)
    
    xgb_train(params, train_dmatrix, eval_dmatrix, './log')
    

if __name__ == '__main__':
    main()