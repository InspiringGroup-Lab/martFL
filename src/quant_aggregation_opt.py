from quant_basic_opt import *
import numpy as np
import torch

K = 22
is_dump = False

def quant_multiply(q_a, s_a, z_a, q_b, s_b, z_b, s_c, z_c):
    """
    q_a: intpoint matrix of a
    s_a: scaling factor of a
    z_a: zero point of a
    q_b: intpoint matrix of b
    s_b: scaling factor of b
    z_b: zero point of b
    s_c: scaling factor of c
    z_c: zero point of c
    C = A * B
    """
    
    print("q_a_shape: ", q_a.shape)
    print("q_b_shape: ", q_b.shape)
    
    q_c = np.zeros((q_a.shape[0],q_b.shape[1]), dtype=np.int64)
    r_c = np.zeros((q_a.shape[0],q_b.shape[1]), dtype=np.int64)
    
    print("q_c_shape: ", q_c.shape)
    print("r_c_shape: ", r_c.shape)
    
    for i in range(q_a.shape[0]):
        for j in range(q_b.shape[1]):
            q_c[i,j] = z_c<<K 
            q_c[i,j] += np.round(s_a*s_b/s_c*2**K*((q_a[i,:]*q_b[:,j]).sum()+z_a*z_b*q_a.shape[1]-z_a*q_b[:,j].sum()-z_b*q_a[i,:].sum()))
            r_c[i,j] = q_c[i,j]%(2**K)
            q_c[i,j] = np.int64(q_c[i,j]>>K) + (r_c[i,j]>>(K-1))
    # print(r_c.min(), r_c.max())
    return q_c

def quant_add(q_a, s_a, z_a, q_b, s_b, z_b, s_c, z_c):
    """
    q_a: intpoint matrix of a
    s_a: scaling factor of a
    z_a: zero point of a
    q_b: intpoint matrix of b
    s_b: scaling factor of b
    z_b: zero point of b
    s_c: scaling factor of c
    z_c: zero point of c
    C = A + B
    """
    print("q_a_shape: ", q_a.shape)
    print("q_b_shape: ", q_b.shape)
    
    q_c = np.zeros(q_a.shape, dtype=np.int64)
    r_c = np.zeros(q_a.shape, dtype=np.int64)
    
    print("q_c_shape: ", q_c.shape)
    print("r_c_shape: ", r_c.shape)
    
    for i in range(q_a.shape[0]):
        for j in range(q_a.shape[1]):
            q_c[i,j] = z_c<<K 
            q_c[i,j] += np.round(s_a/s_c*2**K*q_a[i,j] + s_b/s_c*2**K*q_b[i,j] - s_a/s_c*2**K*z_a - s_b/s_c*2**K*z_b)
            r_c[i,j] = q_c[i,j]%(2**K)
            q_c[i,j] = np.int64(q_c[i,j]>>K) + (r_c[i,j]>>(K-1))
    # print(r_c.min(), r_c.max())
    return q_c
    

def check_quant_multiply_error(floatpoint_array_a, floatpoint_array_b):
    """
    floatpoint_array_a: numpy array of floatpoint numbers a
    floatpoint_array_b: numpy array of floatpoint numbers b
    """
    floatpoint_array_c = np.matmul(floatpoint_array_a,floatpoint_array_b)
    
    scaler_a = calculate_scaling_factor(floatpoint_array_a.min(), floatpoint_array_a.max())
    scaler_b = calculate_scaling_factor(floatpoint_array_b.min(), floatpoint_array_b.max())
    
    zeropoint_a = zero_point(floatpoint_array_a.max(), scaler_a)
    zeropoint_b = zero_point(floatpoint_array_b.max(), scaler_b)
    
    floatpoint_c_min, floatpoint_c_max = calculate_floatpoint_add_mul_range(floatpoint_array_a, floatpoint_array_b, True)
    scaler_c = calculate_scaling_factor(floatpoint_c_min, floatpoint_c_max)
    zeropoint_c = zero_point(floatpoint_c_max, scaler_c)
    
    intpoint_array_a = quant_floatpoint(floatpoint_array_a, scaler_a, zeropoint_a)
    intpoint_array_b = quant_floatpoint(floatpoint_array_b, scaler_b, zeropoint_b)
    intpoint_array_c = quant_multiply(intpoint_array_a, scaler_a, zeropoint_a, intpoint_array_b, scaler_b, zeropoint_b, scaler_c, zeropoint_c)
    
    quant_multiply_error =  floatpoint_array_c - dequant_intpoint(intpoint_array_c, scaler_c, zeropoint_c)
    quant_multiply_error_abs = np.abs(quant_multiply_error)
    print("quantization multiply error: ", quant_multiply_error.min(), quant_multiply_error.max(), quant_multiply_error_abs.mean())
     
    
def check_quant_add_error(floatpoint_array_a, floatpoint_array_b):
    """
    floatpoint_array_a: numpy array of floatpoint numbers a
    floatpoint_array_b: numpy array of floatpoint numbers b
    """
    floatpoint_array_c = np.add(floatpoint_array_a,floatpoint_array_b)
    
    scaler_a = calculate_scaling_factor(floatpoint_array_a.min(), floatpoint_array_a.max())
    scaler_b = calculate_scaling_factor(floatpoint_array_b.min(), floatpoint_array_b.max())
    
    zeropoint_a = zero_point(floatpoint_array_a.max(), scaler_a)
    zeropoint_b = zero_point(floatpoint_array_b.max(), scaler_b)
    
    floatpoint_c_min, floatpoint_c_max = calculate_floatpoint_add_mul_range(floatpoint_array_a, floatpoint_array_b, False)
    scaler_c = calculate_scaling_factor(floatpoint_c_min, floatpoint_c_max)
    zeropoint_c = zero_point(floatpoint_c_max, scaler_c)
    
    intpoint_array_a = quant_floatpoint(floatpoint_array_a, scaler_a, zeropoint_a)
    intpoint_array_b = quant_floatpoint(floatpoint_array_b, scaler_b, zeropoint_b)
    intpoint_array_c = quant_add(intpoint_array_a, scaler_a, zeropoint_a, intpoint_array_b, scaler_b, zeropoint_b, scaler_c, zeropoint_c)
    
    quant_add_error =  floatpoint_array_c - dequant_intpoint(intpoint_array_c, scaler_c, zeropoint_c)
    quant_add_error_abs = np.abs(quant_add_error)
    print("quantization add error: ", quant_add_error.min(), quant_add_error.max(), quant_add_error_abs.mean())
    
def test_check_quant_multiply_error():
    floatpoint_array_a = np.random.rand(1,20)
    floatpoint_array_b = np.random.rand(20,100000)
    check_quant_multiply_error(floatpoint_array_a, floatpoint_array_b)
    
def test_check_quant_add_error():
    floatpoint_array_a = np.random.rand(1,100000)
    floatpoint_array_b = np.random.rand(1,100000)
    check_quant_add_error(floatpoint_array_a, floatpoint_array_b)   


if __name__ == "__main__":
    test_check_quant_multiply_error()
    test_check_quant_add_error()
    