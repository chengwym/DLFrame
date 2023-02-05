import torch
import matplotlib.pyplot as plt
import hyperopt
from hyperopt import hp, fmin, tpe, Trials

import dlframe.infrastructure.pytorch_utils as ptu

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
            
            ###################################
        acc = float(num_correct) / float(num_count)
    return acc

def train(epoches, train_dataloader, eval_dataloader, model, optimizer, criterion, log_dir, plot=True, savemodel=False):
    model.train()
    eval_acc_list = []
    eval_loss_list = []
    train_acc_list = []
    train_loss_list = []
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
        
def fit(params):
    model = 
    criterion = 
    optimizer = 
    train()

def search_hyperparameter(work_dir):
    
    space = {
        
    }
    
    trials = Trials()
    
    best = fmin(
        fn=fit,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )