import torch
from torch import Tensor
from torch import nn
import numpy as np
import random

device = torch.device('cuda')
seed = 3407

def convert_multiple_gpu(model):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
def to_numpy(tensor: Tensor):
    return tensor.to('cpu').detach().numpy()

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def set_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False