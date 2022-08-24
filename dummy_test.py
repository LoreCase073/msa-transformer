import torch
import math
import numpy as np
import torch

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.get_device_name(0))

dir = '/home/lorenzo/Documents/unimg/ML/Data/training_set_Rosetta/training_set'

filetxt = '/list15051.txt'

compr = np.load(dir + '/npz/1a0b_1_A.npz')

c2 = np.load(dir + '/npz/9gaa_1_A.npz')

data = compr.files


print(data)

msa = compr['msa']

msa2 = c2['msa']


print(msa.shape)
print(type(msa))
print(len(msa[0]))

print(msa2.shape)
print(len(msa2[0]))