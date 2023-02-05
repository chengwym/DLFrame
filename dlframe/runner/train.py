import argparse
import torch.nn as nn
import torch.optim as optim

from dlframe.dataprocess.dataloader import *
from dlframe.runner.method import train, check_acc, check_loss, search_hyperparameter
from dlframe.model.mlp import MLP
import dlframe.infrastructure.pytorch_utils as ptu

def main():
    ptu.set_seed(ptu.seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--epoches', type=int, default=10)
    parser.add_argument('--beta1', type=float, default=0.99)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--max_evals', type=int, default=10)
    
    args = parser.parse_args()
    params = vars(args)
    ##########################################
    train_dataloader = DiyDataloader(params['batch_size'], 'train', './')
    test_dataloader = DiyDataloader(params['batch_size'], 'test', './')
    eval_dataloader = DiyDataloader(params['batch_size'], 'eval', './')
    ##########################################
    criterion = nn.MSELoss()
    # model = MLP([93, 64, 32, 16, 1], 5)
    # model.to(ptu.device)
    # optimizer = optim.Adam(model.parameters())
    # train(params['epoches'], train_dataloader, eval_dataloader, model, optimizer, criterion, './log')
    search_hyperparameter(criterion, train_dataloader, eval_dataloader, params, params['work_dir'])

if __name__ == '__main__':
    main()