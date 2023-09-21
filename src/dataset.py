import random
import datetime
import torch
import numpy as np
from scipy.stats import powerlaw
import math
from torchvision import transforms
from bias_distribution import get_bias_distr
from dataset_trec import get_trec_label
from dataset_agnews import get_agnews_label
#from random_distribution_bk import get_rand_distr
# random.seed(111)
# random.seed(222)
# random.seed(333)
# random.seed(444)
# random.seed(555)
seed = datetime.datetime.now()
print("random seed :", seed)
random.seed(seed)
def get_label_index(dataset,label):
    
    if dataset == 'TREC':
        return get_trec_label(label)
    if dataset == 'AGNEWS':
        return get_agnews_label(label)
    else:
        return None

def dataset_output_dim(dataset):
    output_dim = 0
    
    if dataset == 'CIFAR' or dataset == 'FMNIST' or dataset == 'MNIST':
        output_dim = 10
    elif dataset == 'TREC':
        output_dim = 6
    elif dataset == 'AGNEWS':
        output_dim = 4
    else:
        pass
    return output_dim

def samples_split(n_samples, n_participants,split_name = 'uni',root_dataset=100):
    
    sample_nums = [ 0 for _ in range(n_participants)]
    if split_name == 'uni':
        for i in range(n_participants):
            if i == 0:
                sample_nums[i] = root_dataset
            else:
                sample_nums[i] = (n_samples-root_dataset) // n_participants

    elif split_name == 'pow':
        
        sample_nums = powerlaw_size(n_samples-root_dataset, n_participants-1)
        sample_nums.insert(0,root_dataset)
    else:
        print("split_name = ['uni', 'pow']")
        exit(1)

    i = 0
    while sum(sample_nums) < n_samples:
        if i%n_participants == 0:
            i += 1
            continue
        else:
            sample_nums[i%n_participants] += 1
            i += 1

    print("sample_split",sample_nums)
    return sample_nums

def powerlaw_size(sample_num, n_participants,alpha=1.66, res=True):
    
    party_size = sample_num // n_participants

    b = np.linspace(powerlaw.ppf(0.01, alpha), powerlaw.ppf(0.99, alpha), n_participants)
    shard_sizes = list(map(math.ceil, b/sum(b)*party_size*n_participants))
    if res:
        shard_sizes.reverse()
    #print("shard_sizes",shard_sizes)
    return shard_sizes

def split_num(amount,num):
    list1 = []
    for i in range(1,num):
        a = random.randint(0,amount)    
        list1.append(a)
    list1.sort()                        
    list1.append(amount)                

    list2 = []
    for i in range(len(list1)):
        if i == 0:
            b = list1[i]                
        else:
            b = list1[i] - list1[i-1]  
        list2.append(b)

    return list2

def classes_split(dataset,sample_nums, train, n_participants, n_classes, split_name, bias=0):
    
    train_indices = [[] for _ in range(n_participants)]

    if split_name == 'rand':
        data_indices = [[] for _ in range(n_classes)]
        for i,data in enumerate(train):
            if dataset == 'TREC' or dataset == 'SST' or dataset == 'AGNEWS':
                data_indices[get_label_index(dataset,data.label)].append(i)
            else :
                data_indices[data[1]].append(i)

        class_indices = [[i for i in range(n_classes)] for _ in range(n_participants)]
        print('random_class_indices',class_indices)


        sample_sizes = get_bias_distr(dataset,bias)#get_rand_distr(dataset,rand)
        print("sample_sizes",sample_sizes)
        for i in range(len(sample_sizes)):
            for j in range(len(sample_sizes[0])):
                sample_sizes[i][j] = int(sample_sizes[i][j])
                
        print('sample_sizes',sample_sizes)
        
        for i in range(n_participants):
            sample_nums[i] = sum(sample_sizes[i])
            #sample_sizes.append(split_num(sample_nums[i],n_classes)) #每个participant的每个class有几个sample
            needed_samples = sample_nums[i]#这个participant需要的sample总数

            for class_id in class_indices[i]:
                
                if len(data_indices[class_id]) < sample_sizes[i][class_id]:
                    #这个class剩余的sample不足sample_sizes[i]，全部给它
                    selected_num = len(data_indices[class_id])
                else:
                    selected_num = sample_sizes[i][class_id]
                #print('client_id',i,'class_id',class_id,'selected_num',selected_num,'left in this class',len(data_indices[class_id]))
                if selected_num > 0:
                    selected_indices = random.sample(data_indices[class_id], k=selected_num)
                    data_indices[class_id] = list(set(data_indices[class_id])-set(selected_indices))
                    train_indices[i].extend(selected_indices)
                    needed_samples -= selected_num
                #print('train_indices[{}]'.format(i),len(train_indices[i]))
            #print('needed_samples',needed_samples)
            
            overall_needed_samples = needed_samples
            
            class_id = n_classes -1
            while needed_samples > 0 :
                
                left_class = []
                for k in range(n_classes):
                    if len(data_indices[k]) > 0:
                        left_class.append(k)
                        
                #print('left_class',left_class,'class_id',class_id)
                if class_id not in left_class:
                    if class_id < 0:
                        break
                    else:
                        class_id -= 1
                        continue
                
                each_class_need = (overall_needed_samples // len(left_class)) + 1
                if len(data_indices[class_id]) <  each_class_need:
                    selected_num = len(data_indices[class_id])
                else:
                    selected_num =  each_class_need
                #print('client_id',i,'class_id',class_id,'selected_num',selected_num,'left in this class',len(data_indices[class_id]))
                selected_indices = random.sample(data_indices[class_id], k=selected_num)
                data_indices[class_id] = list(set(data_indices[class_id])-set(selected_indices))
                train_indices[i].extend(selected_indices)
                needed_samples -= selected_num
                #print('train_indices[{}]'.format(i),len(train_indices[i]))
                class_id -= 1
                if class_id < 0:
                    break

    elif split_name == 'pow':
        data_indices = [[] for _ in range(n_classes)]
        for i,data in enumerate(train):
            if dataset == 'TREC' or dataset == 'SST':
                data_indices[get_label_index(dataset,data.label)].append(i)
            else :
                data_indices[data[1]].append(i)
        
        class_indices = [[i for i in range(n_classes)] for _ in range(n_participants)]
        print('class_indices',class_indices)

        sample_sizes = [[sample_nums[i] // n_classes  for _ in range(n_classes)] for i in range(n_participants)]#每个participant的每个class有几个sample
        spow_list = powerlaw_size(sample_nums[0], n_classes)
        spow_list.reverse()
        sample_sizes[0] = spow_list
        print('sample_sizes',sample_sizes)

        for i in range(n_participants):
            needed_samples = sample_nums[i]#这个participant需要的sample总数

            for class_id in class_indices[i]:
                
                #sample_sizes[i] participant[i]需要这个class(class_id)的sample数
                if len(data_indices[class_id]) < sample_sizes[i][class_id]:
                    #这个class剩余的sample不足sample_sizes[i]，全部给它
                    selected_num = len(data_indices[class_id])
                else:
                    selected_num = sample_sizes[i][class_id]

                selected_indices = random.sample(data_indices[class_id], k=selected_num)
                data_indices[class_id] = list(set(data_indices[class_id])-set(selected_indices))
                train_indices[i].extend(selected_indices)
                needed_samples -= selected_num

            class_id = 0
            while needed_samples > 0 :
                if len(data_indices[class_id]) <  needed_samples:
                    selected_num = len(data_indices[class_id])
                else:
                    selected_num =  needed_samples
                
                selected_indices = random.sample(data_indices[class_id], k=selected_num)
                data_indices[class_id] = list(set(data_indices[class_id])-set(selected_indices))
                train_indices[i].extend(selected_indices)
                needed_samples -= selected_num
                class_id += 1
                if class_id >= n_classes:
                    break

    elif split_name == 'clb':

        class_per_cli = 2 #每个participant拥有2个class的数据

        #data_indices = [torch.nonzero(train.targets == class_id).view(-1).tolist() for class_id in range(n_classes)]
        data_indices = [[] for _ in range(n_classes)]
        for i,data in enumerate(train):
            data_indices[data[1]].append(i)

        class_indices = [[(class_per_cli*i+j)%n_classes for j in range(class_per_cli)] for i in range(n_participants)]
        class_indices[0] = [i for i in range(n_classes)]
        print('class_indices',class_indices)

        sample_sizes = [sample_nums[i] // class_per_cli for i in range(n_participants)]#每个participant的每个class有几个sample
        sample_sizes[0] = sample_nums[0] // n_classes
        print('sample_sizes',sample_sizes)

        for i in range(n_participants):
            needed_samples = sample_nums[i]#这个participant需要的sample总数

            for class_id in class_indices[i]:
                
                #sample_sizes[i] participant[i]需要这个class(class_id)的sample数
                if len(data_indices[class_id]) < sample_sizes[i]:
                    #这个class剩余的sample不足sample_sizes[i]，全部给它
                    selected_num = len(data_indices[class_id])
                else:
                    selected_num = sample_sizes[i]

                selected_indices = random.sample(data_indices[class_id], k=selected_num)
                data_indices[class_id] = list(set(data_indices[class_id])-set(selected_indices))
                train_indices[i].extend(selected_indices)
                needed_samples -= selected_num

            class_id = 0
            while needed_samples > 0 :
                if len(data_indices[class_id]) <  needed_samples:
                    selected_num = len(data_indices[class_id])
                else:
                    selected_num =  needed_samples
                
                selected_indices = random.sample(data_indices[class_id], k=selected_num)
                data_indices[class_id] = list(set(data_indices[class_id])-set(selected_indices))
                train_indices[i].extend(selected_indices)
                needed_samples -= selected_num
                class_id += 1
                if not class_id < n_classes:
                    break

    elif split_name == 'uni':
        indices = [i for i in range(len(train))]
        random.shuffle(indices)
        split = 0
        for i in range(n_participants):
            train_indices[i].extend(indices[split:split+sample_nums[i]])
            split += sample_nums[i]
                
    else:
        print("Error. split_name = ['uni', 'clb','pow']")
        exit(1)

    print('class_split',[len(train_indices[i]) for i in range(n_participants)])
    return train_indices

def get_data_transform(data: str):
    transform  = transforms.Compose([
            transforms.ToTensor(),
        ])
    if data == 'MNIST' or data == 'FMNIST':
        transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
	        transforms.Resize((28,28)), 
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif data == 'CIFAR':
        transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform

