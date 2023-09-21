from copy import deepcopy
from tkinter import N
import torch
import heapq
import random
import math
import numpy as np
from torch import nn
from train import train_model
from quant_aggregation import *
from model_saver import *
from collections import OrderedDict
from cluster import gap,kmeans
from train import evaluate_model,kappa
from dataset import dataset_output_dim
from homo_encryption import private_model_evaluation
path = '/home/lq/project/fedmarket/quantization_params/'
epoch = 0
cpu_device = torch.device('cpu')
class Aggregator():

    def __init__(self,exp_name,n_participants,n_adversaries,backup_models,client_models,model_structure=None,quantization = False,device=None):
        self.exp_name = exp_name
        self.n_participants = n_participants
        self.n_adversaries = n_adversaries
        self.backup_models = backup_models
        self.client_models = client_models
        self.model_structure = model_structure
        self.device = device
        self.server = 0
        self.quantization = quantization
        self.max_indexes = [0]
    def get_update_gradients(self):
        #计算的是每个local model相对于自己的update
        update_gradients = [compute_update_gradients(self.exp_name,old_model,new_model,self.device) for old_model,new_model in zip(self.backup_models,self.client_models)]
        return update_gradients
    
    def get_update_gradients_toserver(self):
        #计算的是每个local model相对于server的update
        update_gradients_toserver = [compute_update_gradients(self.exp_name,self.backup_models[0],new_model,self.device) for new_model in self.client_models]
        return update_gradients_toserver

    def martFL(self,global_epoch,server_dataloader,loss_fn,change_base,ground_truth_model=None):
        print("change_base :",change_base)
        print("global_epoch :",global_epoch)
        #init
        aggregated_gradient =  [torch.zeros(param.shape).to(self.device) for param in self.model_structure.parameters()]
                
        ##calc_cosine
        clients_update_flattened = [flatten(clip_gradient_update(u,0.01)) for u in self.get_update_gradients()] 
        server_update_flattened = clients_update_flattened[self.server].clone().detach()
        
        # cosine_result = [cosine_xy(server_update_flattened,clients_update_flattened[i]) for i in range(self.n_participants)]
        cosine_result = [encrypted_cosine_xy(server_update_flattened,clients_update_flattened[i]) for i in range(self.n_participants)]
        
        # print('cosine_result',cosine_result)
        cosine_result.pop(self.server)
        if self.server != 0:
            cosine_result.pop(0)
        # print('cosine_result pop server',cosine_result)
        np_cosine_result = np.array(cosine_result) #n-2
        if self.server != 0:
            cosine_result.insert(0,0.0)
        cosine_result.insert(self.server,0.0)
        
        ##cluster (n-1 clients)
        diameter = max(cosine_result)-min(cosine_result)
        print("diameter",diameter)

        n_clusters = gap(np_cosine_result)
        if n_clusters == 1 and diameter > 0.05:
            n_clusters = 2
        
        clusters,centroids = kmeans(np_cosine_result,n_clusters)
        # print('clusters without server',clusters)
        
        if self.server != 0:
            clusters.insert(0,0)
        clusters.insert(self.server,0)#remove server
        # print('clusters insert server',clusters)
        print('centroids : {}'.format(centroids))
        print('{}_clusters : {}'.format(n_clusters,clusters))
        
        center = centroids[-1]
        if n_clusters == 1:
            clusters2 = [1]*self.n_participants
        else:
            clusters2,_ = kmeans(np_cosine_result,2)
            if self.server != 0:
                clusters2.insert(0,0)
            clusters2.insert(self.server,0)#remove server
        
        border = 0
        print('{}_clusters : {}'.format(2,clusters2))
        for i in range(len(clusters2)):
            if i == 0 or i == self.server:
                continue
            if n_clusters == 1 or clusters2[i] != 0:
                cr = cosine_result[i]
                if abs(center-cr) > border:
                    border = abs(center-cr)
        print('border',border)
        
        non_outliers = [1.0 for _ in range(self.n_participants)]

        candidate_server = []

        for i in range(0,self.n_participants):
            if i == 0 or i == self.server:
                # print('remove server: i = {}'.format(i))
                non_outliers[i] = 0.0
                continue
            
            if clusters2[i] == 0 or cosine_result[i] == 0.0:#malicious
                # print('remove attack: i = {}'.format(i))
                non_outliers[i] = 0.0
        
            else: 
                
                dist = abs(center - cosine_result[i])
                non_outliers[i] = 1.0 - dist/(border+1e-6)
                
                if clusters[i] == n_clusters - 1:
                    candidate_server.append(i)
                    non_outliers[i] = 1.0            
            
        ##calc_cosine_weight
        for i in range(self.n_participants):
            cosine_result[i] = non_outliers[i] #*cosine_result[i]
            
        
        cosine_weight = torch.tensor(cosine_result,dtype=torch.float)
        cosine_weight = torch.div(cosine_weight, torch.sum(torch.abs(cosine_weight))+1e-6)
        weight = cosine_weight
        
        # print('final_weight',weight)
        if change_base:
            #random_choose
            all_candidate = []
            low_quality_candidate = []
            random_num = int(0.1*self.n_participants)
            for i,p in enumerate(clusters2):
                if i != 0 and i != self.server:
                    all_candidate.append(i)
                    if p != 0:
                        low_quality_candidate.append(i)
                        
            if len(candidate_server) == 0:
                candidate_server = [i for i in range(self.n_participants) if (i!=0 and i!=self.server)]
            prepare_random = list(set(low_quality_candidate)-set(candidate_server))
            if len(prepare_random) == 0 and len(candidate_server) < 0.5 * self.n_participants:
                prepare_random = list(set(all_candidate)-set(candidate_server))
            print('prepare_random',prepare_random)
            random_candidate = random.sample(prepare_random,min(random_num,len(prepare_random)))
            
                        
            #next_server
            print("This Server: {}".format(self.server))
            print('{} Original Candidate Server : {}'.format(len(candidate_server),candidate_server))
            print('{} Random Candidate Server : {}'.format(len(random_candidate),random_candidate))
            
            candidate_server = list(set(candidate_server)|set(random_candidate))
            candidate_server.sort()
            
            print('{} Final Candidate Server : {}'.format(len(candidate_server),candidate_server))
            
            print("Look for next server:")
            sem = threading.Semaphore(5)
            threads = []
            next_server = 1
            
            for i in candidate_server:
                temp_model = import_model(self.exp_name,self.backup_models[i],self.device)
                temp_model = add_update_to_model(temp_model,self.get_update_gradients()[i],weight=1.0,device=self.device)
                client_thread = MyThread(func=evaluate_model,args=(temp_model, server_dataloader, loss_fn, self.device, 0,dataset_output_dim(self.exp_name.split('-')[0])),semaphore=sem)
                #_,vacc = evaluate_model(temp_model, server_dataloader, loss_fn, self.device, 0)
                #print("Client {}: {}".format(i,vacc))
                client_thread.start()
                threads.append(client_thread)
            
            for t in threads:
                t.join()
            
            score_list = []
            max_score = 0
            for i,t in enumerate(threads):
                score = t.result[2]
                print('Client {}: {}'.format(candidate_server[i],score))
                if score > max_score:
                    max_score = score
                    next_server = candidate_server[i]
                score_list.append(score)
            
            self.server = next_server
            print("Next Server: {}".format(self.server))
        
        # baseline_score
        ground_truth_updates = compute_ground_truth_updates(self.exp_name,self.backup_models[0],ground_truth_model,self.device)
        ground_truth_updates_flattened = flatten(ground_truth_updates)
        baseline_updates_flattened = clients_update_flattened[self.server]
        baseline_score = cosine_xy(ground_truth_updates_flattened,baseline_updates_flattened)
        print('baseline_score',baseline_score)
        #calc_global_update
        update_gradients = self.get_update_gradients()
        for gradient, wt in zip(update_gradients, weight):
            add_gradient_updates(aggregated_gradient, gradient, weight=wt)
        
        if self.quantization:
            backup_model = import_model(self.exp_name,get_backup_name_from_model_name(self.client_models[0]),self.device)
            update_gradients = torch.cat([flatten(u).reshape(1,-1) for u in self.get_update_gradients_toserver()],dim=0)
            updated_model = integrated_quant_aggregation(deepcopy(backup_model),weight,update_gradients,global_epoch)
            for model_name in self.client_models: 
                export_model(self.exp_name,model_name,updated_model)
        else:
            for i,model_name in enumerate(self.client_models):
                model = import_model(self.exp_name,get_backup_name_from_model_name(model_name),self.device)
                updated_model = add_update_to_model(model,aggregated_gradient,weight=1.0,device=self.device)
                export_model(self.exp_name,model_name,updated_model)

        return weight,self.server,baseline_score
    
def compute_ground_truth_updates(exp_name,old_model_name,new_model,device=None):
    old_model = import_model(exp_name,old_model_name,device)
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]
  
def cosine_xy(x,y):
    x = x.clone().detach()
    y = y.clone().detach()
    torch_cosine = torch.cosine_similarity(x,y,dim=0)
    return torch_cosine.item()

def encrypted_cosine_xy(x,y):
    return private_model_evaluation(x,y)

def clip_gradient_update(grad_update, grad_clip):
    clipped_updates = []
    for param in grad_update:
        #param = torch.tensor(param,dtype=torch.float)
        update = torch.clamp(param, min=-grad_clip, max=grad_clip)
        clipped_updates.append(update)
    return clipped_updates     
       
def set_model_parameters(model, unflatten_parameters,device=None):
    if len(unflatten_parameters) == 0 : return model
    if device:
        model = model.to(device)
        unflatten_parameters = [param.to(device) for param in unflatten_parameters]
    
    for param_model, param_update in zip(model.parameters(), unflatten_parameters):
        param_model.data = param_update.data
    
    return model
    
    
def add_update_to_model(model, update, weight=1.0, device=None):
	if len(update) == 0 : return model
	if device:
		model = model.to(device)
		update = [param.to(device) for param in update]
			
	for param_model, param_update in zip(model.parameters(), update):
		param_model.data += weight * param_update.data
	return model


def flatten(grad_update):
    	return torch.cat([update.data.view(-1) for update in grad_update])

def unflatten(flattened, normal_shape):
	grad_update = []
	for param in normal_shape:
		n_params = len(param.view(-1))
		grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
		flattened = flattened[n_params:]

	return grad_update


def compute_update_gradients(exp_name, old_model, new_model, device=None):
    old_model = import_model(exp_name,old_model,device)
    new_model = import_model(exp_name,new_model,device)
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]


def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
	assert len(grad_update_1) == len(
		grad_update_2), "Lengths of the two grad_updates not equal"
	
	for param_1, param_2 in zip(grad_update_1, grad_update_2):
		param_1.data += param_2.data * weight

def add_update_to_model(model, update, weight=1.0, device=None):
	if len(update) == 0 : return model
	if device:
		model = model.to(device)
		update = [param.to(device) for param in update]
			
	for param_model, param_update in zip(model.parameters(), update):
		param_model.data += weight * param_update.data
	return model 
        
        