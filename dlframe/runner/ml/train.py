import argparse

import dlframe.infrastructure.tree_utils as tru
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
    
    args = parser.parse_args()
    params = vars(args)
    
    

if __name__ == '__main__':
    main()