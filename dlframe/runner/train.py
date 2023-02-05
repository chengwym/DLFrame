import argparse
import torch.nn as nn
import torch.optim as optim

from dlframe.dataprocess.dataloader import *
from dlframe.runner.method import train, check_acc, check_loss
from dlframe.model.mlp import MLP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=64)
    ##########################################
    criterion = nn.MSELoss()
    ##########################################
    model = MLP([])
    ##########################################
    optimizer = optim.Adam()
    ##########################################
    

if __name__ == '__main__':
    main()