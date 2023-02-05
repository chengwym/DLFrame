import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

import dlframe.infrastructure.pytorch_utils as ptu
from dlframe.infrastructure.utils import plot_hyper_dict, dict_to_str
from dlframe.model.mlp import MLP

def check_loss(dataloader, model, criterion):
    model.eval()
    with torch.no_grad():
        loss = 0
        for input, target in dataloader:
            input = input.to(ptu.device)
            target = target.to(ptu.device)
            output = model(input)
            _loss = criterion(output, target)
            loss += _loss.item()
    return loss

def check_acc(dataloader, model):
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_count = 0
        for input, target in dataloader:
            input = input.to(ptu.device)
            target = target.to(ptu.device)
            output = model(input)
            # TODO define the evaluation method
            num_correct += (torch.abs(target - output) < 0.1).sum()
            num_count += target.shape[0]
            ###################################
        acc = float(num_correct) / float(num_count)
    return acc

def train(epoches, train_dataloader, eval_dataloader, model, optimizer, criterion, log_dir, plot=True, savemodel=False):
    model.train()
    eval_acc_list = []
    eval_loss_list = []
    train_acc_list = []
    train_loss_list = []
    print('start training.............')
    for epoch in range(epoches):
        train_loss = 0
        for input, target in train_dataloader:
            input = input.to(ptu.device)
            target = target.to(ptu.device)
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'epoch[{epoch}]: loss={train_loss:.6f}')
        eval_loss = check_loss(eval_dataloader, model, criterion)
        eval_acc = check_acc(eval_dataloader, model)
        train_acc = check_acc(train_dataloader, model)
        eval_acc_list.append(eval_acc)
        eval_loss_list.append(eval_loss)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        
    if plot:
        step = range(epoches)
        plt.plot(step, train_acc_list, 'r')
        plt.plot(step, eval_acc_list, 'b')
        plt.xlabel('epoches')
        plt.ylabel('accuracy')
        plt.title('accuracy curve')
        plt.legend(['train acc', 'eval acc'])
        plt.savefig(f'{log_dir}/acc.png')
        plt.figure()
        plt.plot(step, train_loss_list, 'r')
        plt.plot(step, eval_loss_list, 'b')
        plt.xlabel('epoches')
        plt.ylabel('loss')
        plt.title('loss curve')
        plt.legend(['train loss', 'eval loss'])
        plt.savefig(f'{log_dir}/loss.png')
        
    if savemodel:
        torch.save(model.state_dict(), f'{log_dir}/model.pt')

def search_hyperparameter(criterion, train_dataloader, eval_dataloader, params, work_dir):
    
    space = {
        
    }
    
    def fit(p):
        lr = p.get('learning_rate', params['learning_rate'])
        beta1 = p.get('beta1', params['beta1'])
        eps = p.get('eps', params['eps'])
        beta2 = p.get('beta2', params['beta2'])
        weight_decay = p.get('weight_decay', params['weight_decay'])
        
        model = MLP([93, 64, 32, 16, 1], 5)

        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            eps=eps,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
        train(params['epoches'], train_dataloader, eval_dataloader, model, optimizer, criterion, f'{work_dir}'/{dict_to_str(p)})
        
        return {'loss': check_loss(eval_dataloader, model, criterion), 'status': STATUS_OK}
    
    trials = Trials()
    
    best = fmin(
        fn=fit,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    
    plot_hyper_dict(trials.vals, trials.losses())