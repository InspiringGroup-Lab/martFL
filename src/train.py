import torch
import numpy as np
from torch import nn
# from torchtext.legacy.data import Batch
from torchtext.data import Batch
from copy import deepcopy
from model_saver import *
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix,roc_auc_score
from torcheval.metrics.functional import multiclass_f1_score,multiclass_accuracy

def get_loss_function(loss_function_name):
    loss_fn = None
    if loss_function_name == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    if loss_function_name == 'NLL':
        loss_fn = nn.NLLLoss()  
    return loss_fn

def train_model(model, dataloader, loss_fn, device, local_epoch, optim,learning_rate):
    model.train()
    model = model.to(device)
    
    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,eps=1e-4)
    
    for e in range(local_epoch):
        # running local epochs
        #batch = next(iter(dataloader))
        for i,batch in enumerate(dataloader):
        
            if isinstance(batch,Batch):
                data, label = batch.text, batch.label
                data = data.permute(1, 0)
            else:
                data, label = batch[0], batch[1]

            data, label = data.to(device), label.to(device)
            model.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            # print('LEpoch: {}, Loss:{}'.format(e, loss.item()))


    return model


def fine_tuning_model(model, dataloader, loss_fn, device, local_epoch, optim,learning_rate):
    model.train()
    model = model.to(device)
    
    linear_layer = model.fc
    
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    
    if optim == 'SGD':
        optimizer = torch.optim.SGD(linear_layer.parameters(), lr=learning_rate*0.1,momentum=0.9)
    else:
        optimizer = torch.optim.Adam(linear_layer.parameters(), lr=learning_rate*0.1,eps=1e-4)
    
    for e in range(local_epoch):

        for batch in dataloader:
        
            if isinstance(batch,Batch):
                data, label = batch.text, batch.label
                data = data.permute(1, 0)
            else:
                data, label = batch[0], batch[1]

            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('FineTune Epoch: {}, Loss:{}'.format(e, loss.item()))            
    return model

def evaluate_model(model, dataloader, loss_fn, device, global_epoch,num_classes,print_log=False):
    
    model.eval()
    model = model.to(device)
    
    correct_sample = 0.0
    total_sample = 0.0
    total_loss = 0.0
    
    y_true = []
    y_score = []
    y_pred = []
    
    with torch.no_grad():

        for le,batch in enumerate(dataloader):
            
            if isinstance(batch, Batch):
                data, target = batch.text, batch.label
                data = data.permute(1, 0)
            else:
                data, target = batch[0], batch[1]

            data, target = data.to(device), target.to(device)
            pred = model(data)
            
            loss = loss_fn(pred, target)

            correct_sample += (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()
            total_sample += target.size()[0]
            total_loss += loss.item()
            
            y_true.extend(target.data.cpu().numpy())
            y_score.extend(F.softmax(pred,dim=1).data.cpu().numpy())
            y_pred.extend(torch.max(pred, 1)[1].view(target.size()).data.cpu().numpy())
            
            
            # cf_matrix = cf_matrix / np.sum(cf_matrix, axis=1)
            #if i%20 == 0:
            #    print('Eval Loss:{}'.format(loss.item()))


    accuracy =  correct_sample / total_sample
    average_loss = total_loss / (le+1)
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    f1_score = multiclass_f1_score(torch.tensor(y_pred),torch.tensor(y_true),num_classes=num_classes,average='macro').item()
    kp = kappa(cf_matrix)
    # auc_score = roc_auc_score(y_true,y_score,multi_class='ovo')
    
    if print_log:
        print("Epoch: {}. Loss: {:.6f}. Accuracy: {:.4%}.".format(global_epoch + 1, average_loss, accuracy))
    
    return round(average_loss,6), round(accuracy.item(),6),kp,f1_score


def label_flip_train_model(model, dataloader, loss_fn, device, local_epoch, optim,learning_rate, before_label, after_label):
    
    model.train()
    model = model.to(device)
    
    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,eps=1e-4)
    
    for e in range(local_epoch):
        # running local epochs
        for batch in dataloader:
            
            if isinstance(batch, Batch):
                data, label = batch.text, batch.label
                data = data.permute(1, 0)
            else:
                data, label = batch[0], batch[1]
                
            for i,l in enumerate(label):
                if l in before_label:
                    label[i] = after_label[before_label.index(l)]

            data, label = data.to(device), label.to(device)
            model.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

    return model

def label_flip_evaluate_model(model, dataloader, loss_fn, device, global_epoch, before_label, after_label, num_classes,print_log=False):
    
    model.eval()
    model = model.to(device)
    
    correct_sample = 0.0
    total_sample = 0.0
    total_loss = 0.0
    
    
    attack_success = 0.0
    target_total = 0.0
    attack_accuracy = 0.0
    
    y_true = []
    y_score = []
    y_pred = []
    
    with torch.no_grad():

        for le,batch in enumerate(dataloader):
            
            if isinstance(batch, Batch):
                data, target = batch.text, batch.label
                data = data.permute(1, 0)
            else:
                data, target = batch[0], batch[1]

            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = loss_fn(pred, target)

            pred_label = torch.max(pred, 1)[1].view(target.size()).data
            correct_sample += (pred_label == target.data).sum()
            total_sample += target.size()[0]
            total_loss += loss.item()
            
            this_target_total,this_attack_success = calc_attack_accuracy(pred_label,target,before_label, after_label)
            target_total += this_target_total
            attack_success += this_attack_success
            
            y_true.extend(target.data.cpu().numpy())
            y_score.extend(F.softmax(pred,dim=1).data.cpu().numpy())
            y_pred.extend(torch.max(pred, 1)[1].view(target.size()).data.cpu().numpy())
            
            
    accuracy =  correct_sample / total_sample
    average_loss = total_loss / le
    attack_accuracy = attack_success / target_total
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    f1_score = multiclass_f1_score(torch.tensor(y_pred),torch.tensor(y_true),num_classes=num_classes,average='macro').item()
    kp = kappa(cf_matrix)
    
    if print_log:
        print("Epoch: {}. Loss: {:.6f}. Accuracy: {:.4%}. Attack: {:.4%}".format(global_epoch+1,average_loss, accuracy,attack_accuracy))
        
    return round(average_loss,6), round(accuracy.item(),6),round(attack_accuracy,6),kp,f1_score

def backdoor_train_model(model, dataloader, loss_fn, device, local_epoch, optim,learning_rate, before_label, after_label,alpha_loss = 0.95):
    
    backup_model = deepcopy(model)

    target_params_variables = dict()
    for name, param in backup_model.named_parameters():
        target_params_variables[name] = backup_model.state_dict()[name].clone().detach().requires_grad_(False)

    model.train()
    model = model.to(device)
    
    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,eps=1e-4)
    
    for e in range(local_epoch):
        # running local epochs
        for batch in dataloader:
            if isinstance(batch, Batch):
                data, label = batch.text, batch.label
                data = data.permute(1, 0)
            else:
                data, label = batch[0], batch[1]

            for i,l in enumerate(label):
                if l in before_label:
                    label[i] = after_label[before_label.index(l)]

            data, label = data.to(device), label.to(device)
            model.zero_grad()
            pred = model(data)
            class_loss = loss_fn(pred, label)
            
            distance_loss = model_dist_norm_var(model, target_params_variables)
            loss = alpha_loss * class_loss + (1 - alpha_loss) * distance_loss
            loss.backward()
            optimizer.step()

    return model

def calc_attack_accuracy(pred,target,before_label,after_label):
    #print('pred',pred)
    #print('target',target)
    label_flip_dict = {}
    for i in range(len(before_label)):
        label_flip_dict[before_label[i]] = after_label[i]
    #print(label_flip_dict)
    target_total = 0
    attack_success = 0
    indices = []
    for i,t in enumerate(target):
        if t.item() in label_flip_dict.keys():
            target_total += 1
            indices.append(i)
    for idx in indices:
        if pred[idx] == label_flip_dict[target[idx].item()]:
            attack_success += 1
    #print(target_total,attack_success)
    return target_total,attack_success

def model_dist_norm_var(model, target_params_variables, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.cuda.FloatTensor(size).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
        layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)

def BenignTraining(exp_name,client_model_name,dataloader,loss_fn,device,local_epoch,optimizier,learning_rate):
    client_model = import_model(exp_name,client_model_name,device)
    client_model = train_model(client_model, dataloader,loss_fn,device,local_epoch,optimizier,learning_rate)
    return export_model(exp_name,client_model_name,client_model)

def MaliciousTraining(exp_name,attack,client_model_name,dataloader,loss_fn,device,local_epoch,optimizier,learning_rate,before_label=None,after_label=None):
    client_model = import_model(exp_name,client_model_name,device)

    if attack == 'free_rider':
        pass    
    elif attack == 'label_flipping':
        client_model = label_flip_train_model(client_model,dataloader,
        loss_fn,device,local_epoch,optimizier, 
        learning_rate,before_label,after_label)
    
    elif attack == 'rescaling':
        client_model = train_model(client_model,dataloader,
        loss_fn, device, local_epoch, optimizier, learning_rate)
        noise = 5
        for grad in client_model.parameters():
            grad.data *= (noise * 2) * torch.rand(size=grad.shape, device=grad.device) - noise
            grad.data *= noise
    
    elif attack == 'sign_randomizing':
        client_model = train_model(client_model,dataloader,
        loss_fn, device, local_epoch, optimizier, learning_rate)
        # randomly flip the signs of the elements in gradient, element-wise
        for grad in client_model.parameters():
            grad.data *= torch.tensor(np.random.choice([-1.0, 1.0], size=grad.shape),dtype=torch.float).to(device)

    elif attack == 'param_ramdomizing':
        for grad in client_model.parameters():
            grad.data = (torch.rand(grad.size()) * 2 -1).to(device)
            
    elif attack == 'backdoor':
        client_model = backdoor_train_model(client_model,dataloader,
        loss_fn,device,local_epoch,optimizier, 
        learning_rate,before_label,after_label,alpha_loss=0.8)

    elif attack == 'sybil':
        #client_id = get_client_id(client_model_name)
        first_malicious_client = get_model_name(1)
        client_model = import_model(exp_name,first_malicious_client,device)

    return export_model(exp_name,client_model_name,client_model)

def kappa(confusion_matrix):
    """
    计算多分类混淆矩阵的kappa系数 
    """
    N = np.sum(confusion_matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(confusion_matrix[0])):
        sum_po += confusion_matrix[i][i]
        i_row = np.sum(confusion_matrix[i, :])
        i_col = np.sum(confusion_matrix[:, i]) 
        sum_pe += i_row * i_col
    po = sum_po / N
    pe = sum_pe / (N * N)
    kia = (po - pe) / (1 - pe)
    return round(kia,2)