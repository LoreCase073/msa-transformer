import math
from turtle import forward
from typing import Optional
from sympy import symmetrize
import torch
from torch import embedding
from torch import narrow
import torch.nn as nn
import torch.nn.functional as F



class PositionalEmbedding(nn.Embedding):
    """Learns positional embedding for a fixed size. Padding ids are ignored"""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_pos = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input of [bsz x seqlen]"""
        if input.size(1)>self.max_pos:
            raise ValueError(f"Sequence too long!"
            f"Maximum sequence length of {self.max_pos}")
        #TODO: what is input.ne? Computes if input != other element-wise
        #Calcola element-wise not equal!
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask)*mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )