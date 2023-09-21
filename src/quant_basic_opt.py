import numpy as np

Q_MAX = 4096
Q_MIN = 0
eps = 0.5
is_dump = False
dump_path = "quantized_data/"


def floatpoint_minimum_maximum(floatpoint_array):
    """
    floatpoint_array: numpy array of floatpoint numbers
    """
    return floatpoint_array.min(), floatpoint_array.max()


def calculate_scaling_factor(floatpoint_min, floatpoint_max):
    """
    floatpoint_min: minimum value of floatpoint numbers
    floatpoint_max: maximum value of floatpoint numbers
    """
    
    q_max = Q_MAX
    q_min = Q_MIN
    
    r_min = min(floatpoint_min, 0) - eps
    r_max = max(floatpoint_max, 0) + eps
    
    r_max_min_range = r_max - r_min
    
    scaler = r_max_min_range / (q_max-q_min)
    
    return scaler

def zero_point(floatpoint_max, scaler):
    """
    floatpoint_max: maximum value of floatpoint numbers
    scaler: scaling factor
    """
    r_max = max(floatpoint_max, 0) + eps
    q_max = Q_MAX
    zeropoint = np.round(q_max - r_max/scaler)
    return np.int32(zeropoint)

def quant_floatpoint(floatpoint_array, scaler, zeropoint):
    """
    floatpoint_array: numpy array of floatpoint numbers
    """
    return np.round(floatpoint_array/scaler + zeropoint)

def dequant_intpoint(intpoint_array, scaler, zeropoint):
    """
    intpoint_array: numpy array of intpoint numbers
    """
    return (intpoint_array-zeropoint)*scaler

def check_quantization_error(floatpoint_array, scaler, zeropoint):
    """
    floatpoint_array: numpy array of floatpoint numbers
    scaler: scaling factor
    zeropoint: zero point of intpoint numbers
    """
    return floatpoint_array - dequant_intpoint(quant_floatpoint(floatpoint_array, scaler, zeropoint), scaler, zeropoint)

def calculate_floatpoint_add_mul_range(floatpoint_array_a,floatpoint_array_b, is_mul = False):
    """
    floatpoint_array_a : numpy array of floatpoint numbers a
    floatpoint_array_b : numpy array of floatpoint numbers b
    is_multiply (bool, optional): a+b or a*b. Defaults to False.
    """

    if is_mul: # a * b
        floatpoint_min = min(floatpoint_array_a.min()*floatpoint_array_b.min(), floatpoint_array_a.min()*floatpoint_array_b.max(), floatpoint_array_a.max()*floatpoint_array_b.min(), floatpoint_array_a.max()*floatpoint_array_b.max())
        floatpoint_max = max(floatpoint_array_a.min()*floatpoint_array_b.min(), floatpoint_array_a.min()*floatpoint_array_b.max(), floatpoint_array_a.max()*floatpoint_array_b.min(), floatpoint_array_a.max()*floatpoint_array_b.max())
    else: # a + b
        floatpoint_min = min(floatpoint_array_a.min(), floatpoint_array_b.min())
        floatpoint_max = max(floatpoint_array_a.max(), floatpoint_array_b.max())
        
    return floatpoint_min, floatpoint_max

def calculate_floatpoint_add_mul_range_(fp_a_min, fp_a_max, fp_b_min, fp_b_max, is_mul = False):
    """
    fp_a_min : minimum value of floatpoint numbers a
    fp_a_max : maximum value of floatpoint numbers a
    fp_b_min : minimum value of floatpoint numbers b
    fp_b_max : maximum value of floatpoint numbers b
    is_multiply (bool, optional): a+b or a*b. Defaults to False.
    """
    if is_mul: # a * b
        floatpoint_min = min(fp_a_min*fp_b_min, fp_a_min*fp_b_max, fp_a_max*fp_b_min, fp_a_max*fp_b_max)
        floatpoint_max = max(fp_a_min*fp_b_min, fp_a_min*fp_b_max, fp_a_max*fp_b_min, fp_a_max*fp_b_max)
    else: # a + b
        floatpoint_min = min(fp_a_min, fp_b_min)
        floatpoint_max = max(fp_a_max, fp_b_max)
        
    return floatpoint_min, floatpoint_max

def test_calculate_quantization_error():
    floatpoint_array = np.random.randn(1000)
    print(floatpoint_array.min(), floatpoint_array.max())
    scaler = calculate_scaling_factor(floatpoint_minimum_maximum(floatpoint_array)[0], floatpoint_minimum_maximum(floatpoint_array)[1])
    zeropoint = zero_point(floatpoint_minimum_maximum(floatpoint_array)[1], scaler)
    quant_error = check_quantization_error(floatpoint_array, scaler, zeropoint)
    print("quantization error: ", quant_error.min(), quant_error.max())
   
   
def dump_txt(q, z, s, prefix):
    np.savetxt(prefix+"_q.txt", q.flatten(), fmt='%u', delimiter=',')
    # print(z, s)
    f1 = open(prefix+"_z.txt", 'w+')
    if(str(z)[0] == '['):
        f1.write(str(z)[1:-1])
    else:
        f1.write(str(z))
    f1.close()
    f2 = open(prefix+"_s.txt", 'w+')
    if(str(s)[0]=='['):
        f2.write(str(s)[1:-1])
    else:
        f2.write(str(s))
    f2.close()
     
if __name__ == "__main__":
    test_calculate_quantization_error()
    