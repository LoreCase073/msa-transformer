from typing import Sequence, TypeVar
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


TensorLike = TypeVar("TensorLike", np.ndarray, torch.Tensor)


class MSADataset(Dataset):

    def __init__(self, file_csv, npz, max_seq_len, max_pos, padding_idx):
        self.file_csv = pd.read_csv(file_csv)
        self.npz = npz
        self.max_seq_len = max_seq_len
        self.max_pos = max_pos

        self.padding_idx = padding_idx

        self.resize_transf = ResizeMSA(self.max_seq_len, self.max_pos)
        self.binarize_transf = ContactMapBinarization(8)
        

    def __len__(self):
        return self.file_csv.shape[0]
 
    def __getitem__(self, index):
        file = np.load(os.path.join(self.npz, self.file_csv.iloc[index, 1]+'.npz'))
        name = self.file_csv.iloc[index, 1]
        msa = torch.from_numpy(file['msa'])
        dist = torch.from_numpy(file['dist6d'])
    
        msa, dist = self.resize_transf(msa, dist)
         
        
        distance = self.binarize_transf(dist)
        

        return {"msa":msa, "distance":distance, "name":name}




class ContactMapBinarization(object):
    def __init__(self, 
        dist, 
    ): 
        self.dist = dist

    def __call__(self, dist):
        map = dist

        binarized = torch.where((map>0)&(map<=self.dist),1,0)
        

        return binarized


class ResizeMSA(object):
    def __init__(self,  
        max_seq_len,
        max_pos,
    ): 
        self.max_seq = max_seq_len
        self.max_pos = max_pos


    def __call__(self, msa, dist):
        seq = msa
        d = dist

        if seq.size(0) > self.max_seq:
            seq = seq[0:self.max_seq,:]
        if seq.size(1) > self.max_pos:
            seq = seq[:,0:self.max_pos]
            d = d[0:self.max_pos,0:self.max_pos]


        return seq, d


def collate_tens(
    sequences: Sequence[TensorLike], 
    constant_value=0, 
    dtype=None,
) -> TensorLike:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    


    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

def collate_tensors(batch):
    msa = [i['msa'] for i in batch]
    distance = [i['distance'] for i in batch]
    name = [i['name'] for i in batch]

    

    
    
    msa_b = collate_tens(msa, 21)
    distance_b = collate_tens(distance,0)

    msa_b = msa_b.int()

    return msa_b, distance_b, name