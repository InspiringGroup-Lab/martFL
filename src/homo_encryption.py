# import sys
# sys.path.append(r'/home/lq/project/fedmarket_github/SEAL-Python')   # 路径为存放so文件的文件夹路径

import seal
import numpy
import torch
from torch.linalg import norm
import math

eps = 1e-8

def private_model_evaluation(model_a_flatten_parameters,model_b_flatten_parameters,sloth = 4096):
    
    ckks_context, scale, public_key,secret_key, relin_keys, gal_keys, encoder, encryptor, decryptor,evaluator = key_generater()

    norm_parameters_a = norm(model_a_flatten_parameters)
    norm_parameters_b = norm(model_b_flatten_parameters)
    
    party_a_parameters = (model_a_flatten_parameters / norm_parameters_a).detach().cpu().numpy()
    party_b_parameters = (model_b_flatten_parameters / norm_parameters_b).detach().cpu().numpy()
    # print(len(party_a_parameters),len(party_b_parameters))
    
    ciphertext = encrypt_model(party_a_parameters,encryptor,encoder,scale,sloth = sloth)
    ciphertext = multiply_plain(ciphertext,party_b_parameters,evaluator,encoder,scale,sloth = sloth)
    cosine_similarity = decrypt_model(ciphertext,encoder,decryptor,sloth = sloth)
    return cosine_similarity

def key_generater():
    ckks_params = seal.EncryptionParameters(seal.scheme_type.ckks)
    poly_modulus_degree = 8192
    ckks_params.set_poly_modulus_degree(poly_modulus_degree)
    ckks_params.set_coeff_modulus(seal.CoeffModulus.Create(poly_modulus_degree, (60, 40, 40, 60)))
    scale = math.pow(2.0, 40)
    ckks_context = seal.SEALContext(ckks_params)
    keygen = seal.KeyGenerator(ckks_context)
    secret_key = keygen.secret_key()
    public_key = keygen.create_public_key()
    relin_keys = keygen.create_relin_keys()
    gal_keys = keygen.create_galois_keys()
    encoder = seal.CKKSEncoder(ckks_context)
    encryptor = seal.Encryptor(ckks_context, public_key)
    decryptor = seal.Decryptor(ckks_context, secret_key)
    evaluator = seal.Evaluator(ckks_context)

    return ckks_context, scale, public_key,secret_key, relin_keys, gal_keys, encoder, encryptor, decryptor,evaluator

def encrypt_model(parameters,encryptor,encoder,scale,sloth = 4096):
    count = math.ceil(len(parameters)/sloth)
    # print('encrypt_model','count',count)

    ciphertext = [encryptor.encrypt(encoder.encode(parameters[i*sloth:(i+1)*sloth],scale)) for i in range(count)]
    
    return ciphertext

def multiply_plain(ciphertext,parameters,evaluator,encoder,scale,sloth = 4096):
    count = math.ceil(len(parameters)/sloth)
    # print('multiply_plain','count',count)
    #print('ciphertext',len(ciphertext))
    ret = []
    for i in range(count):
        plaintext = parameters[i*sloth:(i+1)*sloth]
        plaintext = encoder.encode(plaintext,scale)
        original_ciphertext = ciphertext[i]
        evaluator.multiply_plain_inplace(original_ciphertext,plaintext)
        ret.append(original_ciphertext)
    return ret

def decrypt_model(ciphertext,encoder,decryptor,sloth = 4096):
    count = len(ciphertext)
    # print('decrypt_model','count',count)
    result = 0
    for i in range(count):
        plaintext = numpy.array(encoder.decode(decryptor.decrypt(ciphertext[i])))
        result += plaintext.sum()
    return result

def test_private_model_evaluation():
    model_a = torch.rand(100000)
    model_b = torch.rand(100000)
    # print(private_model_evaluation(model_a,model_b))
    
if __name__ == '__main__':
    test_private_model_evaluation()