import torch
from torch.utils.data import Dataset
# from torchtext.legacy import data,datasets
from torchtext import data,datasets

class myTrec(Dataset):
    def __init__(self):

        self.text_field = data.Field(lower=True)
        self.label_field = data.LabelField(dtype = torch.long, sequential=False)

    def get_dataset(self):
        #train_data, test_data=datasets.TREC.splits(self.TEXT,self.LABEL,fine_grained=False)
        train_data = datasets.TREC('../datasets/train_5500.label',self.text_field,self.label_field,fine_grained=False)
        test_data = datasets.TREC('../datasets/TREC_10.label',self.text_field,self.label_field,fine_grained=False)
        
        #MAX_VOCAB_SIZE = 25_000
        #self.TEXT.build_vocab(train_data,max_size=MAX_VOCAB_SIZE,vectors="../datasets/glove.6B.300d",unk_init=torch.Tensor.normal_)
        #self.LABEL.build_vocab(train_data)
        print('TREC Dataset Len: {}'.format(len(train_data)))
        print('TREC Dataset Len: {}'.format(len(test_data)))
        return train_data, test_data

def get_trec_label(label):
    dct = {'HUM': 0, 'ENTY': 1, 'DESC': 2, 'NUM': 3, 'LOC': 4, 'ABBR': 5}
    if label in dct.keys():
        return dct[label]
    else:
        return 0
    


    

