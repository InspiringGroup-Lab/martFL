import os
import time
import threading
from copy import deepcopy
from watcher import Watcher
from mythread import *
from config import *
from model import *
from participant import *
from dataset import *
from train import *
from aggregator import *
from model_saver import import_model,export_model,backup_model
from model_saver import backup_models as bk_models
from result_saver import *
import warnings
warnings.filterwarnings('ignore')
from early_stopping import EarlyStopping


def main():

    Watcher()

    Args = Config()

    Exp_Name = Args.dataset + '-' + Args.model_name + '-' + \
        str(Args.n_participant) + '-' + Args.attack + '-' + \
        str(Args.n_adversary) + '-' + Args.sample_split + '-' + \
        Args.class_split + '-' + Args.aggregator + '-' + str(Args.quantization)

    if 'martFL' in Args.aggregator:
        Exp_Name = Exp_Name + '-' + str(Args.change_base)

    if Args.class_split == 'rand':
        Exp_Name = Exp_Name + '-' + str(Args.alpha)

    Exp_Name = Exp_Name + '-' + str(int(time.time()))
    print('Exp_Name:' + Exp_Name)
    print('change_base:',str(Args.change_base))


    device = torch.device(Args.device)
    attack = Args.attack
    if attack is not None:
        if Args.dataset == 'MNIST':
            before_label = [1,7]
            after_label = [7,1]
        elif Args.dataset == 'FMNIST' :
            before_label = [7,9]
            after_label = [9,7]
        elif Args.dataset == 'CIFAR':
            before_label = [1,9]
            after_label = [9,1]
        else:
            before_label = [0,1]
            after_label = [1,0]
    
    n_participants = Args.n_participant
    n_adversaries = Args.n_adversary
    learning_rate = Args.learning_rate

    # Prepare Data
    print('# Prepare Data.')
    participants = Participant(np=n_participants,
                               dataset=Args.dataset,
                               batch_size=Args.batch_size,
                               output_dim=dataset_output_dim(Args.dataset),
                               root_dataset=Args.root_dataset,
                               sample_split=Args.sample_split,
                               class_split=Args.class_split,bias=Args.alpha,
                               server_sample_ratio=Args.server_ratio,
                               device=Args.device)

    train_dataloaders = participants.train_dataloaders
    test_dataloader = participants.test_dataloader

    print('Data Distribution:')
    for i,data_distr in enumerate(participants.data_distribution):
        print('Client:',i , data_distr, sum(data_distr))
    
    participants.clean_dataset()

    Model = get_model(Args.model_name)
    
    if Args.model_name == 'TextCNN':
        INPUT_DIM = participants.embedding['embed_num']
        LABEL_DIM = participants.embedding['class_num']
        PAD_IDX   = participants.embedding['pad_token']
        model_structure = Model(INPUT_DIM,LABEL_DIM,PAD_IDX)
    else:
        model_structure = Model()
    
    client_models = [get_model_name(cid) for cid in range(n_participants)]
    for cmodel in client_models:
        export_model(Exp_Name,cmodel,deepcopy(model_structure))

    #backup_models = [backup_model(Exp_Name,cmodel_name) for cmodel_name in client_models]
    backup_models = bk_models(Exp_Name,client_models)
    aggregator_name = Args.aggregator
    aggregator = Aggregator(Exp_Name,n_participants,n_adversaries,backup_models,client_models,model_structure,Args.quantization,device)
    

    filename = save_info(Exp_Name,Args.learning_rate,Args.optimizer,
                           Args.loss_fn,Args.batch_size,
                           Args.global_epoch,Args.local_epoch,participants.data_distribution,Args.__dict__)
    print('Exp_INFO:',filename)
    
    total_loss = []
    total_acc = []
    total_attack_acc = []
    total_kp = []
    total_f1 = []
    total_baselines = []
    total_baselines_scores = []
    
    early_stopping = EarlyStopping(patience=100, verbose=True, delta=0, 
        path=MODEL_PATH + Exp_Name +'/' + 'BestGlobalModel')
    
    for global_epoch in range(Args.global_epoch):
        
        backup_models = bk_models(Exp_Name,client_models)

        #Threads
        sem_num = Args.semaphore
        sem = threading.Semaphore(sem_num)
        threads = []
        
        # Federate Training
        print('# Federated Training. Epoch: {}. LR: {}.'.format(global_epoch+1,learning_rate))
        for cid in range(n_participants):
            
            # Normal User 
            if cid < n_participants-n_adversaries:
                client_thread = MyThread(func=BenignTraining,args=(Exp_Name,client_models[cid], train_dataloaders[cid],
                get_loss_function(Args.loss_fn), device, Args.local_epoch, Args.optimizer, learning_rate),semaphore=sem)
            # Malicious User
            else:
                print('## Malicious Client {} is Training. Attack = {}.'.format(cid,attack))
                if attack == 'sybil' and cid == n_participants-n_adversaries: 
                    client_model = import_model(Exp_Name,client_models[cid],device)
                    client_model = label_flip_train_model(client_model,train_dataloaders[cid],
                    get_loss_function(Args.loss_fn),device,Args.local_epoch, Args.optimizer,
                    learning_rate,before_label,after_label)
                    export_model(Exp_Name,client_models[cid],client_model)
                else:
                    client_thread = MyThread(func=MaliciousTraining,args=(Exp_Name,attack,client_models[cid], train_dataloaders[cid],
                    get_loss_function(Args.loss_fn), device, Args.local_epoch, Args.optimizer, learning_rate,before_label,after_label),semaphore=sem)
            
            if attack == 'sybil' and cid > n_participants-n_adversaries:
                pass
            else:
                client_thread.start()
                threads.append(client_thread)
        
        for t in threads:
            t.join()
               
        #Federated Aggregating
        print('# Federated Aggregating. Aggregator: {}'.format(aggregator_name))    
            
        # ground_truth_baseline
        previous_global_model = import_model(Exp_Name,backup_models[0],device)
        ground_truth_model = train_model(previous_global_model,test_dataloader,
            get_loss_function(Args.loss_fn),device,Args.local_epoch, Args.optimizer, learning_rate)
        
        aggregate_weight, chosen_baseline,baseline_score = aggregator.martFL(global_epoch=global_epoch,
        server_dataloader=test_dataloader,loss_fn=get_loss_function(Args.loss_fn),change_base = Args.change_base,ground_truth_model=ground_truth_model)
        total_baselines.append(chosen_baseline)
        total_baselines_scores.append(baseline_score)
        
        
        print('# Federated Evaluating.')
        global_model = import_model(Exp_Name,client_models[0],device)
        if attack == 'label_flipping' or attack == 'sybil' or attack == 'backdoor':
            loss,acc,atk_acc,kp,f1 = label_flip_evaluate_model(global_model,test_dataloader, get_loss_function(Args.loss_fn), device,global_epoch,before_label,after_label,dataset_output_dim(Args.dataset),print_log=True)
        else:    
            loss,acc,kp,f1 = evaluate_model(global_model, test_dataloader, get_loss_function(Args.loss_fn), device, global_epoch,dataset_output_dim(Args.dataset),print_log=True)
            atk_acc = 0.0

        total_loss.append(loss)
        total_acc.append(acc)
        total_attack_acc.append(atk_acc)
        total_kp.append(kp)
        total_f1.append(f1)

        #Save Epoch Result
        print('# Save Result:',end=' ')
        filename = save_epoch_result(Exp_Name,loss,acc,atk_acc,aggregate_weight.tolist(),aggregate_weight.tolist() ,global_epoch+1,kp,f1)
        print(filename)
        
        #Early Stopping
        early_stopping(acc, global_model)
        if early_stopping.early_stop:
            break
    
    ft_result = None 
    if Args.fine_tuning:
        print("Fine_tuning",Args.fine_tuning)
        best_model = deepcopy(global_model)
        best_model.load_state_dict(torch.load(MODEL_PATH + Exp_Name +'/' + 'BestGlobalModel'))
        finetune_model = best_model.to(device)
        
        total_ft_loss = []
        total_ft_acc = []
        total_ft_attack_acc = []
        total_ft_kp = []
        total_ft_f1 = []
        
        for ft_epoch in range(Args.fine_tuning_epoch):
            finetune_model = fine_tuning_model(finetune_model, train_dataloaders[0],get_loss_function(Args.loss_fn) , device, 2, Args.optimizer, learning_rate)
            if attack == 'label_flipping' or attack == 'sybil' or attack == 'backdoor':
                ft_loss,ft_acc,ft_atk_acc,ft_kp,ft_f1 = label_flip_evaluate_model(finetune_model,test_dataloader, get_loss_function(Args.loss_fn), device,ft_epoch,before_label,after_label,dataset_output_dim(Args.dataset),print_log=True)
            else:   
                ft_loss,ft_acc,ft_kp,ft_f1 = evaluate_model(finetune_model, test_dataloader, get_loss_function(Args.loss_fn), device, ft_epoch,dataset_output_dim(Args.dataset),print_log=True)  
                ft_atk_acc = 0.0
            
            
            total_ft_loss.append(ft_loss)
            total_ft_acc.append(ft_acc)
            total_ft_attack_acc.append(ft_atk_acc)
            total_ft_kp.append(ft_kp)
            total_ft_f1.append(ft_f1)
            
        for i,model_name in enumerate(client_models):
            export_model(Exp_Name,model_name,finetune_model)
            
        ft_result = {
            'loss':total_ft_loss,
            'acc':total_ft_acc,
            'atk_acc':total_ft_attack_acc,
            'kp':total_ft_kp,
            'f1':total_ft_f1
        }
            
    #Save Result
    print('# Save Result:',end=' ')
    filename = save_result(Exp_Name,total_loss,total_acc,total_attack_acc,total_kp,total_f1,total_baselines,total_baselines_scores,ft_result)
    print(filename)
    clean_model(Exp_Name,n_participants)


if __name__ == '__main__':
    main()
