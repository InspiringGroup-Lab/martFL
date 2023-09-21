import os
import json
from shutil import copystat
def save_info(exp_name,lr, optim, loss_fn, bz, ge, le, sample_num,args):
    path = '../save_result/{}/'.format(exp_name)
    if not os.path.exists(path):
        os.makedirs(path) 
    file_name = path + 'info.txt'
    text = {
        "exp_name":exp_name,
        "learning_rate":lr,
        "optimizer":optim,
        "loss_fn":loss_fn,
        "batch_size":bz,
        "global_epoch":ge,
        "local_epoch":le,
        "sample_num":sample_num,
        "args":json.dumps(args)
    }
    text_json = json.dumps(text)
    f = open(file_name, 'w')
    f.write(text_json)
    f.close()
    return file_name 


def save_epoch_result(exp_name,loss,acc,atk_acc,agg_weight,cost,global_epoch,kp,f1):
    path = '../save_result/{}/'.format(exp_name)
    if not os.path.exists(path):
        os.makedirs(path) 
    file_name = path + 'epoch-{}.txt'.format(global_epoch)
    text = {
        "exp_name":exp_name,
        "loss":loss,
        "acc":acc,
        "atk_acc":atk_acc,
        "agg_weight":agg_weight,
        "cost":cost,
        "kappa":kp,
        "f1":f1,
    }
    text_json = json.dumps(text)
    f = open(file_name, 'w')
    f.write(text_json)
    f.close()
    return file_name 

def save_result(exp_name,loss,acc,atk,kp,f1,baseline=None,baseline_score=None,ft_result=None):
    path = '../save_result/{}/'.format(exp_name)
    if not os.path.exists(path):
        os.makedirs(path) 
    file_name = path + exp_name + '-Final.txt'
    text = {
        "exp_name":exp_name,
        "loss":loss,
        "acc":acc,
        "atk":atk,
        "kappa":kp,
        "f1":f1,
    }
    if baseline is not None:
        text['baseline'] = baseline
        text['baseline_score'] = baseline_score
    if ft_result is not None:
        text['ft_result'] = ft_result
    text_json = json.dumps(text)
    f = open(file_name, 'w')
    f.write(text_json)
    f.close()
    return file_name 
