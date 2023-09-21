import numpy as np
import random
from dataset import powerlaw_size
from scipy.stats import powerlaw
random.seed(233)

dataset_distr = {'mnist':[5923.0, 6742.0, 5958.0, 6131.0, 5842.0, 5421.0, 5918.0, 6265.0, 5851.0, 5949.0],
                'fmnist':[6000.0, 6000.0, 6000.0, 6000.0, 6000.0, 6000.0, 6000.0, 6000.0, 6000.0, 6000.0],
                'cifar':[5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0],
                'trec':[1223.0, 1250.0, 1162.0, 896.0, 835.0, 86.0],
                'agnews':[3007.0, 3030.0, 2965.0, 2997.0],
                'sst':[3310.0, 1624.0, 3610.0]
}


def ramdom_split(amount,num):
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

def root_generator(alpha,n_sample,n_class):
    b = powerlaw_size(n_sample,n_class,alpha,True)
    b.reverse()
    return b

def unbias_split(n_class,num_sample,dataset):
    ratio = [dataset_distr[dataset][i]/sum(dataset_distr[dataset]) for i in range(n_class)]
    return [int(num_sample*ratio[i]) for i in range(n_class)]

def random_bias_split(n_class,have_class,num_sample,dataset):
    unbiased_dataset = unbias_split(n_class,num_sample,dataset)
    # print(unbiased_dataset)
    # print(list(range(n_class)),n_class-have_class)
    droped = random.sample(list(range(n_class)),n_class-have_class)
    for i in droped:
        unbiased_dataset[i] = 5
    return unbiased_dataset

def bias_split(n_class,have_class,num_sample,dataset):
    sum_samples = 0
    choose_class = []
    for i in range(have_class):
        this_class = BIAS_CLASS[dataset][i]
        choose_class.append(this_class)
        sum_samples += dataset_distr[dataset][this_class]
    ratio = [round(dataset_distr[dataset][i]/sum_samples,2) if i in choose_class else 0 for i in range(n_class)]
    # print('ratio',ratio)
    return [int(num_sample*ratio[i]) for i in range(n_class)]

def bias_split_low_quality(n_class,have_class,num_sample,dataset):
    sum_samples = 0
    choose_class = []
    for i in range(have_class):
        this_class = BIAS_CLASS[dataset][i]
        choose_class.append(this_class)
        sum_samples += dataset_distr[dataset][this_class]
    # ratio = [round(dataset_distr[dataset][i]/sum_samples,2) if i in choose_class else 0 for i in range(n_class)]
    ratio = [round(have_class*dataset_distr[dataset][i]/(sum_samples*n_class),2) if i in choose_class else 0 for i in range(n_class)]
    
    #print('ratio',ratio)
    return [int(num_sample*ratio[i]) if i in choose_class else 2 for i in range(n_class)]

DISTR = {#"mnist":[50,60000,10,100],
        "fmnist":[30,60000,10,200],
        # "emnist":[20,124800,26,200],
        "cifar":[30,50000,10,200],
        "trec":[20,5500,6,120],
        "agnews":[20,12000,4,200],
        #"sst":[20,8544,3,300]
        }

BIAS_CLASS = {
    "mnist":[1,7,0,2,3,4,5,6,8,9],
    "fmnist":[2,4,0,1,3,5,6,7,8,9],
    "cifar":[1,9,0,2,3,4,5,6,7,8],
    "trec":[0,1,2,3,4,5],
    "agnews":[0,1,2,3],
    "sst":[0,1,2]
}


#dataset = 'mnist'
#dataset = 'fmnist'
#dataset = 'trec'
#dataset = 'cifar'
def generate_bias_dataset():
    bias_ratio = [0.3] 
    server_bias = False
    have_class_ratio = 0.5
    like_server = True
    for dataset in ['trec','agnews','fmnist','cifar']:
        for id,percent in enumerate(bias_ratio):

            n_participants = DISTR[dataset][0]
            n_bias = int(percent*n_participants)
            sample_nums = DISTR[dataset][1]
            n_class = DISTR[dataset][2]
            have_class = int(have_class_ratio * n_class)
            root_dataset = DISTR[dataset][3]
            sample_size = [root_dataset if i == 0 else ((sample_nums-root_dataset)//n_participants) for i in range(n_participants)]
            final_distribution = []

            for i in range(n_participants):
                p_num =  sample_size[i]
                if i == 0: #root
                    if server_bias:
                        final_distribution.append(bias_split(n_class,have_class,root_dataset,dataset))
                    else:
                        final_distribution.append(unbias_split(n_class,root_dataset,dataset))
                    
                        
                elif i < n_participants-n_bias: #high-quality
                    # print('#high-quality',i)
                    final_distribution.append(unbias_split(n_class,p_num,dataset))
                else: #low-quality
                    # print('low-quality',i)
                    if like_server:
                        final_distribution.append(bias_split_low_quality(n_class,have_class,p_num,dataset))
                    else:
                        final_distribution.append(ramdom_split(p_num,n_class)) 
            print('{}[{}]='.format(dataset,id),end='')
            print(final_distribution)

def generate_random_dataset():
    server_bias = False
    have_class_ratio = 0.5
    for dataset in ['fmnist']:
        n_participants = DISTR[dataset][0]
        sample_nums = DISTR[dataset][1]
        n_class = DISTR[dataset][2]
        have_class = int(have_class_ratio * n_class)
        root_dataset = DISTR[dataset][3]
        sample_size = [root_dataset if i == 0 else ((sample_nums-root_dataset)//n_participants) for i in range(n_participants)]
        final_distribution = []

        for i in range(n_participants):
            p_num =  sample_size[i]
            if i == 0: #root
                if server_bias:
                    final_distribution.append(bias_split(n_class,have_class,root_dataset,dataset))
                else:
                    final_distribution.append(unbias_split(n_class,root_dataset,dataset))    
            else: #low-quality
                final_distribution.append(ramdom_split(p_num,n_class)) 
                
        print('{}[{}]='.format(dataset,5),end='')
        print(final_distribution)
    
def generate_inclusive_dataset():
    bias_ratio = 0.30
    malicious_ratio = 0.40
    server_bias = False
    have_class_ratio = 0.5
    like_server = False
    id = 28
    for dataset in ['trec','fmnist','agnews','cifar']:
        n_participants = DISTR[dataset][0]
        n_bias = int(bias_ratio*n_participants)
        n_malicious = int(malicious_ratio*n_participants)
        sample_nums = DISTR[dataset][1]
        n_class = DISTR[dataset][2]
        have_class = int(have_class_ratio * n_class)
        root_dataset = DISTR[dataset][3]
        sample_size = [root_dataset if i == 0 else ((sample_nums-root_dataset)//n_participants) for i in range(n_participants)]
        final_distribution = []

        for i in range(n_participants):
            p_num =  sample_size[i]
            if i == 0: #root
                if server_bias:
                    final_distribution.append(bias_split(n_class,have_class,root_dataset,dataset))
                else:
                    final_distribution.append(unbias_split(n_class,root_dataset,dataset))
                
                    
            elif i < n_participants-n_bias-n_malicious: #high-quality
                # print('high-quality',i)
                final_distribution.append(unbias_split(n_class,p_num,dataset))
            elif i < n_participants-n_malicious: #low-quality
                # print('low-quality',i)
                if like_server:
                    final_distribution.append(bias_split_low_quality(n_class,have_class,p_num,dataset))
                else:
                    # final_distribution.append(ramdom_split(p_num,n_class)) 
                    final_distribution.append(random_bias_split(n_class,have_class,p_num,dataset))
            else:
                # print('malicious',i)
                final_distribution.append(unbias_split(n_class,p_num,dataset))
        print('{}[{}]='.format(dataset,id),end='')
        print(final_distribution)
        
        
def generate_random_biased_dataset():
    bias_ratio = 0.3
    malicious_ratio = 0
    server_bias = False
    have_class_ratio = 0.5
    # like_server = True
    id = 25
    for dataset in ['trec','fmnist','agnews','cifar']:
        n_participants = DISTR[dataset][0]
        n_bias = int(bias_ratio*n_participants)
        n_malicious = int(malicious_ratio*n_participants)
        sample_nums = DISTR[dataset][1]
        n_class = DISTR[dataset][2]
        have_class = int(have_class_ratio * n_class)
        root_dataset = DISTR[dataset][3]
        sample_size = [root_dataset if i == 0 else ((sample_nums-root_dataset)//n_participants) for i in range(n_participants)]
        final_distribution = []
        
        for i in range(n_participants):
            p_num =  sample_size[i]
            if i == 0: #root
                if server_bias:
                    final_distribution.append(bias_split(n_class,have_class,root_dataset,dataset))
                else:
                    final_distribution.append(unbias_split(n_class,root_dataset,dataset))
                
                    
            elif i < n_participants-n_bias-n_malicious: #high-quality
                # print('high-quality',i)
                final_distribution.append(unbias_split(n_class,p_num,dataset))
            elif i < n_participants-n_malicious: #low-quality
                # print('low-quality',i)    
                final_distribution.append(random_bias_split(n_class,have_class,p_num,dataset)) 
            else:
                # print('malicious',i)
                final_distribution.append(unbias_split(n_class,p_num,dataset))

        
        print('{}[{}]='.format(dataset,id),end='')
        print(final_distribution)

def generate_biased_pow_dataset():
    bias_ratio = [0.2,0.3,0.4] 
    server_bias = True
    have_class_ratio = 0.5
    like_server = True
    for dataset in ['trec','agnews','fmnist','cifar']:
        
        n_participants = DISTR[dataset][0]
        root_dataset = DISTR[dataset][3]
        sample_nums = DISTR[dataset][1]
        n_class = DISTR[dataset][2]
        
        pow_dataset = powerlaw_size(sample_nums-root_dataset, n_participants-1)
        
        for id,percent in enumerate(bias_ratio):

            n_bias = int(percent*n_participants)
            have_class = int(have_class_ratio * n_class)
            
            
            sample_size = [root_dataset if i == 0 else pow_dataset[i-1] for i in range(n_participants)]
            final_distribution = []

            for i in range(n_participants):
                p_num =  sample_size[i]
                if i == 0: #root
                    if server_bias:
                        final_distribution.append(bias_split(n_class,have_class,root_dataset,dataset))
                    else:
                        final_distribution.append(unbias_split(n_class,root_dataset,dataset))
                    
                        
                elif i < n_participants-n_bias: #high-quality
                    # print('#high-quality',i)
                    final_distribution.append(unbias_split(n_class,p_num,dataset))
                else: #low-quality
                    # print('low-quality',i)
                    if like_server:
                        final_distribution.append(bias_split_low_quality(n_class,have_class,p_num,dataset))
                    else:
                        final_distribution.append(ramdom_split(p_num,n_class)) 
            print('{}[{}]='.format(dataset,id),end='')
            print(final_distribution)
        
# generate_random_dataset()
# generate_bias_dataset()
# generate_inclusive_dataset()
# generate_random_biased_dataset()
# generate_biased_pow_dataset()
