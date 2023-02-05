from torch.utils.data import DataLoader, Dataset

class DiyDataset(Dataset):
    def __init__(
        self,
        mode,
        path
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
        if self.mode == 'test':
            return self.data[index]
        else:
            return self.data[index], self.target[index]
    
    def __len__(self):
        return self.data.shape[0]
    
def DiyDataloader(batch_size, mode, path):
    dataset = DiyDataset(mode, path)
    dataloader = DataLoader(
        dataset,
        batch_size,
        False if mode=='test' else True,
    )
    return dataloader