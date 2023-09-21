import gc
from torchtext.data import Dataset,Batch,BucketIterator
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST,FashionMNIST,CIFAR10,SVHN,EMNIST
from dataset import *
from partition import Partition
from dataset_trec import myTrec
from dataset_agnews import myAGNEWS

class Participant:
    
    def __init__(self,np,dataset,batch_size,output_dim,root_dataset,
                 sample_split = 'uni',class_split = 'uni',bias=0,
                 server_sample_ratio=0.5,device=torch.device('cuda:0')):
        
        self.root = '../datasets'
        self.np = np
        self.dataset = dataset
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.server_sample_ratio = server_sample_ratio
        self.sample_split = sample_split
        self.class_split = class_split
        self.device = device
        self.bias = bias
        self.data_distribution = [[] for i in range(self.np)]
        self.embedding = {}
        self.root_dataset = root_dataset
        
        if self.dataset == 'MNIST':
            self.task = 'image_classification'
            self.train_datasets, self.test_dataset, self.server_test_dataset = self.prepare_mnist_dataset()
        elif self.dataset == 'FMNIST':
            self.task = 'image_classification'
            self.train_datasets, self.test_dataset, self.server_test_dataset = self.prepare_fmnist_dataset()
        elif self.dataset == 'CIFAR':
            self.task = 'image_classification'
            self.train_datasets, self.test_dataset, self.server_test_dataset = self.prepare_cifar_dataset()
        elif self.dataset == 'TREC':
            self.task = 'text_classification'
            self.train_datasets, self.test_dataset, self.server_test_dataset = self.prepare_trec_dataset()
        elif self.dataset == 'AGNEWS':
            self.task = 'text_classification'
            self.train_datasets, self.test_dataset, self.server_test_dataset = self.prepare_agnews_dataset()
        
        else:
            print('ERROR! Dataset = [MNIST,FMNIST,CIFAR,TREC,AGNEWS]')

        for i,dataset in enumerate(self.train_datasets):
            distr = [0.0 for _ in range(self.output_dim)]
            for j in range(len(dataset)):
                if self.dataset in ['TREC','AGNEWS']:
                    if self.dataset == 'TREC':
                        dic = {'HUM': 0, 'ENTY': 1, 'DESC': 2, 'NUM': 3, 'LOC': 4, 'ABBR': 5}
                    if self.dataset == 'AGNEWS':
                        dic = {'4': 0, '1': 1, '2': 2, '3': 3}
                    distr[dic[dataset[j].label]] += 1
                else:
                    distr[dataset[j][1]] += 1
            self.data_distribution[i] = distr

        self.shard_sizes = [len(self.train_datasets[i]) for i in range(self.np)]
        if self.dataset in ['TREC','AGNEWS']:
            
            self.train_dataloaders = [BucketIterator(train_dataset, batch_size=self.batch_size, device=self.device, sort_key=lambda x: len(x.text),train=True) for train_dataset in self.train_datasets]
            self.test_dataloader = BucketIterator(self.test_dataset, batch_size = self.batch_size, sort_key=lambda x: len(x.text), device=self.device)
            self.server_test_dataloader = BucketIterator(self.server_test_dataset, batch_size = self.batch_size, sort_key=lambda x: len(x.text), device=self.device)

        else:
            self.train_dataloaders = [DataLoader(self.train_datasets[i],batch_size=self.batch_size) for i in range(self.np)]
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size)
            self.server_test_dataloader = DataLoader(self.server_test_dataset, batch_size=self.batch_size)
            
    def clean_dataset(self):
        del self.train_datasets
        del self.test_dataset
        del self.server_test_dataset
        gc.collect()

    def get_embedding(self):
        return self.embedding
     
    def prepare_mnist_dataset(self):
        
        train_datasets = [[] for _ in range(self.np)]
        transform = get_data_transform('MNIST')
        train_data = MNIST(root=self.root,train=True,transform=transform,download=True)
        test_data  = MNIST(root=self.root,train=False,transform=transform,download=True)
        
        n_samples = len(train_data)
        n_classes = self.output_dim

        train_indices = spliting('MNIST',n_samples,self.np,n_classes,self.sample_split,self.class_split,train_data,self.bias,self.server_sample_ratio,self.root_dataset)
        
        for i in range(self.np):
            train_datasets[i] = Partition(train_data,train_indices[i])

        server_test_dataset = train_datasets[0]#Partition(train_data,server_test_indices)

        return train_datasets, test_data, server_test_dataset

    def prepare_fmnist_dataset(self):
        train_datasets = [[] for _ in range(self.np)]
        transform = get_data_transform('FMNIST')
        train_data = FashionMNIST(root=self.root,train=True,transform=transform,download=True)
        test_data  = FashionMNIST(root=self.root,train=False,transform=transform,download=True)
        #print_data_distr('FMNIST',train_data,10)
        #train_data = myFMNIST(train=True,transform=transform)
        #test_data = myFMNIST(train=False,transform=transform)
        
        n_samples = len(train_data)
        n_classes = self.output_dim

        train_indices = spliting('FMNIST',n_samples,self.np,n_classes,self.sample_split,self.class_split,train_data,self.bias,self.server_sample_ratio,self.root_dataset)
        
        for i in range(self.np):
            train_datasets[i] = Partition(train_data,train_indices[i])

        server_test_dataset =  train_datasets[0] #Partition(train_data,server_test_indices)

        return train_datasets, test_data, server_test_dataset

    def prepare_cifar_dataset(self):
        train_datasets = [[] for _ in range(self.np)]
        transform = get_data_transform('CIFAR')
        train_data = CIFAR10(root=self.root,train=True,transform=transform,download=True)
        test_data  = CIFAR10(root=self.root,train=False,transform=transform,download=True)
        #print_data_distr('CIFAR',train_data,10)
        #train_data = myCIFAR(train=True,transform=transform)
        #test_data = myCIFAR(train=False,transform=transform)
        
        n_samples = len(train_data)
        n_classes = self.output_dim

        train_indices = spliting('CIFAR',n_samples,self.np,n_classes,self.sample_split,self.class_split,train_data,self.bias,self.server_sample_ratio,self.root_dataset)
        
        for i in range(self.np):
            train_datasets[i] = Partition(train_data,train_indices[i])

        server_test_dataset = train_datasets[0] #Partition(train_data,server_test_indices)

        return train_datasets, test_data, server_test_dataset

    def prepare_trec_dataset(self):
        train_datasets = [[] for _ in range(self.np)]
        
        trec = myTrec()

        train_data,test_data = trec.get_dataset()
        
        n_samples = len(train_data)
        n_classes = self.output_dim

        train_indices = spliting_text('TREC',n_samples,self.np,n_classes,self.sample_split,self.class_split,train_data,self.bias,self.root_dataset)
        
        fields = [('text', trec.text_field), ('label', trec.label_field)]
        train_datas = split_torchtext_dataset_classes(train_data, train_indices,self.server_sample_ratio)
        train_datasets = []
        for data in train_datas:
            dataset = Dataset(data,fields)
            train_datasets.append(dataset)
        server_dataset = train_datasets[0]#Dataset(server_data,fields)
        trec.text_field.build_vocab(*(train_datasets + [test_data]))
        trec.label_field.build_vocab(*(train_datasets + [test_data]))

        self.embedding['embed_num'] = len(trec.text_field.vocab)
        self.embedding['class_num'] = len(trec.label_field.vocab)
        self.embedding['pad_token'] = trec.text_field.vocab.stoi[trec.text_field.pad_token]

        return train_datasets, test_data, server_dataset
    def prepare_agnews_dataset(self):
        
        train_datasets = [[] for _ in range(self.np)]
        
        agnews = myAGNEWS()

        train_data,test_data = agnews.get_dataset()
        #print_data_distr('AGNEWS',train_data,4)
        n_samples = len(train_data)
        n_classes = self.output_dim

        train_indices = spliting_text('AGNEWS',n_samples,self.np,n_classes,self.sample_split,self.class_split,train_data,self.bias,self.root_dataset)
        
        fields = [('label', agnews.label_field),("title", None),('text', agnews.text_field)]
        train_datas = split_torchtext_dataset_classes(train_data, train_indices,self.server_sample_ratio)
        train_datasets = []
        for data in train_datas:
            dataset = Dataset(data,fields)
            train_datasets.append(dataset)
        server_dataset = train_datasets[0]#Dataset(server_data,fields)
        agnews.text_field.build_vocab(*(train_datasets + [test_data]))
        agnews.label_field.build_vocab(*(train_datasets + [test_data]))

        self.embedding['embed_num'] = len(agnews.text_field.vocab)
        self.embedding['class_num'] = len(agnews.label_field.vocab)
        self.embedding['pad_token'] = agnews.text_field.vocab.stoi[agnews.text_field.pad_token]

        return train_datasets, test_data, server_dataset

def spliting_text(dataset,n_samples,n_participants,n_class,sample_split,class_split,train_data,bias,root_dataset):
    sample_nums = samples_split(n_samples, n_participants, split_name = sample_split,root_dataset=root_dataset)
        
    train_indices = classes_split(dataset, sample_nums, train_data, n_participants, n_class, split_name = class_split,bias=bias)

    return train_indices#, server_test_indices

def spliting(dataset,n_samples,n_participants,n_class,sample_split,class_split,train_data,bias,server_sample_ratio=0.5,root_dataset=100):
    sample_nums = samples_split(n_samples, n_participants, split_name = sample_split,root_dataset=root_dataset)
        
    train_indices = classes_split(dataset,sample_nums, train_data, n_participants, n_class, split_name = class_split,bias=bias)

    return train_indices #, server_test_indices

def split_torchtext_dataset_classes(examples,train_indices,server_test_ratio):
    train_datasets = []

    for client_indices in train_indices:
        dataset = [examples[i] for i in client_indices]
        train_datasets.append(dataset)


    return train_datasets#,server_test
