from torch.utils.data import DataLoader, Dataset

import dlframe.dataprocess.read as rd

class DiyDataset(Dataset):
    def __init__(
        self,
        mode,
        path
    ):
        self.mode = mode
        if mode=='test':
            self.data = rd.read_csv(path, mode)
        elif mode=='train':
            self.data, self.target = rd.read_csv(path, mode)
        elif mode=='eval':
            self.data, self.target = rd.read_csv(path, mode)
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