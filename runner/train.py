import argparse
import torch.nn as nn
import torch.optim as optim

from dataprocess.dataloader import *
from runner.method import train, check_acc, check_loss
from model.mlp import MLP

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