import os
import torch
import shutil
import threading
from mythread import MyThread
MODEL_PATH = '../models/'

def get_model_name(cid):
    return 'CModel{}'.format(cid)

def get_client_id(model_name):
    return int(model_name.replace('CModel', ''))

def get_model_name_from_backup(backup_name):
    return backup_name[:-2]

def get_backup_name_from_model_name(model_name):
    return model_name+'bk'

def export_model(exp_name,model_name,model):
    path = MODEL_PATH+exp_name +'/'
    if not os.path.exists(path):
        os.makedirs(path) 
    torch.save(model, path+model_name)
    #print('Update Model: {}'.format(model_name))
    

def backup_model(exp_name,model_name):
    path = MODEL_PATH+exp_name +'/'
    shutil.copy(path+model_name,path+model_name+'bk')
    #print('Backup Model: {}'.format(model_name))
    return model_name+'bk'

def backup_models(exp_name, models_list,sem_num=20):
    sem = threading.Semaphore(sem_num)
    threads = []
    bk_model_name = []

    path = MODEL_PATH+exp_name +'/'
    for model_name in models_list:
        bk_thread = MyThread(shutil.copy,(path+model_name,path+model_name+'bk'),sem)
        bk_thread.start()
        threads.append(bk_thread)
        bk_model_name.append(model_name+'bk')
        
    for t in threads:
        t.join()
        
    #print('Backup Model: {}'.format(bk_model_name))
    return bk_model_name

def import_model(exp_name,model_name,device):
    path = MODEL_PATH+exp_name +'/'
    model = torch.load(path+model_name)
    model = model.to(device)
    return model

def clean_model(exp_name,n_participants):
    path = MODEL_PATH+exp_name +'/'
    # os.remove(MODEL_PATH+exp_name +'/'+'CModel0bk')
    for i in range(n_participants):
        os.remove(MODEL_PATH+exp_name +'/'+'CModel'+str(i))
        os.remove(MODEL_PATH+exp_name +'/'+'CModel'+str(i)+'bk')
    print('Delete Models: {}.'.format(path))
