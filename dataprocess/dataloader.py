from torch.utils.data import DataLoader, Dataset

class Cheng_Dataset(Dataset):
    def __init__(
        self,
        mode
    ):
        self.mode = mode
        if mode=='test':
            pass
        elif mode=='train':
            pass
        elif mode=='eval':
            pass
        else:
            raise ValueError(f'The mode {mode} is uncorrect. It should be test, train, or eval')
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass
    
def Cheng_Dataloader(batch_size, mode):
    dataset = Cheng_Dataset()
    dataloader = DataLoader(
        dataset,
        batch_size,
        False if mode=='test' else True,
    )
    return dataloader