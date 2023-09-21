import torch
import random
from torchtext.vocab import Vectors
#from torch.utils.data import Dataset
from torchtext import data,datasets
from torchtext.data import TabularDataset,Dataset
#from torchtext.legacy import data,datasets
#from torchtext.legacy.data import TabularDataset,Dataset

class myAGNEWS(Dataset):
    def __init__(self):

        self.text_field = data.Field(lower=True)
        self.label_field = data.LabelField(dtype = torch.long, sequential=False)
    
    def get_dataset(self):

        train_path = '../datasets/train.csv'
        test_path = '../datasets/test.csv'
        train_fields = [("label", self.label_field), ("title", None), ("text", self.text_field)]
        train_data = TabularDataset(path=train_path, format="csv", fields=train_fields, skip_header=True)
        train_data = random.sample(list(train_data),int(len(train_data)*0.1))
        train_data = Dataset(train_data,train_fields)
        test_fields = [("label", self.label_field), ("title", None), ("text", self.text_field)]
        test_data = TabularDataset(path=test_path, format="csv", fields=test_fields, skip_header=True)
        test_data =random.sample(list(test_data),int(len(test_data)*0.1))
        test_data = Dataset(test_data,test_fields)
        vectors = Vectors('../datasets/glove.6B.300d.txt')
        self.text_field.build_vocab(train_data, vectors=vectors)

        return train_data,test_data
        
        

def get_agnews_label(label):
    dct = {'0':0, '1': 1, '2': 2, '3': 3,'4': 0}
    if label in dct.keys():
        return dct[label]
    else:
        return None
    

def read_csv(file_name):
    f = open(file_name, 'r')
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    for row in rows:
        final_list.append([row.split(',')[2],int(row.split(',')[0])%4])
    return final_list
    

