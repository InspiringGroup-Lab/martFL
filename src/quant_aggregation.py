from quant_aggregation_opt import quant_multiply,quant_add
from quant_basic_opt import *
import numpy as np
import torch

def from_torch_to_numpy(tensor):
    """
    tensor: torch tensor
    """
    return tensor.cpu().detach().numpy()

def from_numpy_to_torch(array):
    """
    array: numpy array
    """
    return torch.from_numpy(array).cuda()

def quant_aggregation(k,u, epoch):
    """
    k: aggregation_weight, shape = [1*n]
    u: updates for each client, shape = [n*m]
    u_prime = k * u
    """

    k_array = from_torch_to_numpy(k).reshape(1,-1)
    u_array = from_torch_to_numpy(u)
    print("k_array_shape: ", k_array.shape)
    print("u_array_shape: ", u_array.shape)
    
    scaler_k = calculate_scaling_factor(floatpoint_minimum_maximum(k_array)[0], floatpoint_minimum_maximum(k_array)[1])
    scaler_u = calculate_scaling_factor(floatpoint_minimum_maximum(u_array)[0], floatpoint_minimum_maximum(u_array)[1])
    
    zero_point_k = zero_point(floatpoint_minimum_maximum(k_array)[1], scaler_k)
    zero_point_u = zero_point(floatpoint_minimum_maximum(u_array)[1], scaler_u)
    
    u_prime_min, u_prime_max = calculate_floatpoint_add_mul_range(k_array, u_array, is_mul = True)
    scaler_u_prime = calculate_scaling_factor(u_prime_min, u_prime_max)
    zero_point_u_prime = zero_point(u_prime_max, scaler_u_prime)
    
    q_k = quant_floatpoint(k_array, scaler_k, zero_point_k)
    q_u = quant_floatpoint(u_array, scaler_u, zero_point_u)
    q_u_prime = quant_multiply(q_k, scaler_k, zero_point_k, q_u, scaler_u, zero_point_u, scaler_u_prime, zero_point_u_prime)
    
    if is_dump == True:
        dump_txt(q_k,zero_point_k,scaler_k, dump_path+'epoch{}_k'.format(epoch))
        dump_txt(np.transpose(q_u),zero_point_u,scaler_u, dump_path+'epoch{}_u'.format(epoch))
        dump_txt(q_u_prime,zero_point_u_prime,scaler_u_prime, dump_path+'epoch{}_u_prime'.format(epoch))
        
    return q_u_prime, scaler_u_prime, zero_point_u_prime, u_prime_min, u_prime_max


def quant_update(w_t,q_u_prime, scaler_u_prime, zero_point_u_prime, u_prime_min, u_prime_max, epoch):
    """
    w_t: current model, shape = [1*d]
    u_prime: aggregated updates, shape = [1*d]
    q_u_prime: quantized aggregated updates, shape = [1*d]
    scaler_u_prime: scaling factor of q_u_prime
    zero_point_u_prime: zero point of q_u_prime
    w_t_plus_1 = w_t + u_prime
    """

    w_t_array = from_torch_to_numpy(torch.cat([w.view(-1) for w in w_t.parameters()]).reshape(1,-1))
    print("w_t_array_shape: ", w_t_array.shape)
    
    scaler_w_t = calculate_scaling_factor(floatpoint_minimum_maximum(w_t_array)[0], floatpoint_minimum_maximum(w_t_array)[1])
    zero_point_w_t = zero_point(floatpoint_minimum_maximum(w_t_array)[1], scaler_w_t)
    
    w_t_plus_1_min, w_t_plus_1_max = calculate_floatpoint_add_mul_range_(floatpoint_minimum_maximum(w_t_array)[0], 
                                                                         floatpoint_minimum_maximum(w_t_array)[1], 
                                                                         u_prime_min, u_prime_max, is_mul = False)
    
    scaler_w_t_plus_1 = calculate_scaling_factor(w_t_plus_1_min, w_t_plus_1_max)
    zero_point_w_t_plus_1 = zero_point(w_t_plus_1_max, scaler_w_t_plus_1)
    
    q_w_t = quant_floatpoint(w_t_array, scaler_w_t, zero_point_w_t)
    q_w_t_plus_1 = quant_add(q_w_t, scaler_w_t, zero_point_w_t, q_u_prime, scaler_u_prime, zero_point_u_prime, scaler_w_t_plus_1, zero_point_w_t_plus_1)
    
    if is_dump:
        dump_txt(q_w_t,zero_point_w_t,scaler_w_t, dump_path+'epoch{}_w_t'.format(epoch))
        dump_txt(q_w_t_plus_1,zero_point_w_t_plus_1,scaler_w_t_plus_1, dump_path+'epoch{}_w_t_plus_1'.format(epoch))

    return q_w_t_plus_1, scaler_w_t_plus_1, zero_point_w_t_plus_1, w_t_plus_1_min, w_t_plus_1_max

def unflatten_model_parameters(model, flattened_parameters):
    """
    model: pytorch model
    flattened_parameters: flattened parameters of the model
    """
    flattened_parameters = flattened_parameters.reshape(-1)
    unflattened_model = []
    for param in model.parameters():
        n_params = len(param.view(-1))
        unflattened_model.append(from_numpy_to_torch(flattened_parameters[:n_params]).reshape(param.size()).float())
        flattened_parameters = flattened_parameters[n_params:]
    
    return unflattened_model

def set_model_parameters(model, unflatten_parameters):
    """
    model: pytorch model
    unflatten_parameters: unflattened parameters of the model
    """
    for param, unflatten_param in zip(model.parameters(), unflatten_parameters):
        param.data = unflatten_param.data
    return model
    

def integrated_quant_aggregation(w_t,k,u,epoch):
    """
    w_t: current model, shape = [1*m]
    k: aggregation_weight, shape = [1*n]
    u: updates for each client, shape = [n*m]
    w_t_plus_1 = w_t + k * u
    """
    
    q_u_prime, scaler_u_prime, zero_point_u_prime, u_prime_min, u_prime_max = quant_aggregation(k,u,epoch)
    q_w_t_plus_1, scaler_w_t_plus_1, zero_point_w_t_plus_1, _, _ = quant_update(w_t,q_u_prime, scaler_u_prime, zero_point_u_prime, u_prime_min, u_prime_max, epoch)
    fp_q_w_t_plus_1_array = dequant_intpoint(q_w_t_plus_1, scaler_w_t_plus_1, zero_point_w_t_plus_1)
    q_w_t_plus_1 = set_model_parameters(w_t, unflatten_model_parameters(w_t, fp_q_w_t_plus_1_array))
    return q_w_t_plus_1
    