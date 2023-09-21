from torch.utils.data import Dataset
#from torchtext.legacy import data
from torchtext import data

class Partition(Dataset):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        try:
            self.fields = data.fields
        except:
            pass
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]
    

