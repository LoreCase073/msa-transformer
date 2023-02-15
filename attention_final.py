import math
from turtle import forward
from typing import Optional
import torch
from torch import embedding
from torch import narrow
import torch.nn as nn
import torch.nn.functional as F

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


class AxialTL(nn.Module):
    """Class that defines the axial transformer layer, the MSA transformer block"""

    def __init__(
        self,
        embedding_dim: int = 384,
        ffnet_embedding_dim: int = 768,
        num_att_heads: int = 12,
        dropout: float = 0.2,
        attention_dropout: float = 0.2,
        activation_dropout: float = 0.2,
        max_tokens: int = 2**13,
    ) -> None:
        super().__init__()


        #Inizializzo i parametri
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.ffnet_embedding_dim = ffnet_embedding_dim
        self.num_att_heads = num_att_heads
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_tokens = max_tokens


        row_att = RowSelfAttention(
            self.embedding_dim,
            self.num_att_heads,
            dropout = self.dropout,
            max_tokens = self.max_tokens
        )

        col_att = ColumnSelfAttention(
            self.embedding_dim,
            self.num_att_heads,
            self.dropout,
            self.max_tokens,
        )

        ffl = FeedForwardNet(
            self.embedding_dim,
            self.ffnet_embedding_dim,
            self.dropout,
            self.max_tokens,
        )

        self.row_attention = self.res_net(row_att)
        self.col_attention = self.res_net(col_att)
        self.ffl = self.res_net(ffl)

    def res_net(self, layer: nn.Module):
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
            self.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        x, row_attention = self.row_attention(x, padding_mask)
        x, col_attention = self.col_attention(x, padding_mask)
        x = self.ffl(x)

        if need_weights:
            return x, col_attention, row_attention
        else:
            return x
        


#FeedForwardNet
#feedforward network, GELU activation function and 2 fully connected layers, with dropout
class FeedForwardNet(nn.Module):
    def __init__(self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        dropout: float = 0.2,
        max_tokens: int = 2 ** 13,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens = max_tokens
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embedding_dim, self.ffn_embedding_dim)
        self.fc2 = nn.Linear(self.ffn_embedding_dim, self.embedding_dim)

    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class NormalizedResidualBlock(nn.Module):
    def __init__(self,
        layer: nn.Module,
        embedding_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        #TODO: implementare layer normalization
        self.layer_norm = LayerNorm(self.embedding_dim)


    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        output = self.layer(x, *args, **kwargs)
        if isinstance(output, tuple):
            x, *out = output
        else:
            x = output
            out = None
        
        x = self.dropout(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x






###########################################################


#CPH, simmetrizzazione, apc, calcolo della regressione logistica su features....


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

#Logistic Regression on attention heads
class ContPredictionHead(nn.Module):

    def __init__(self,
        in_feat : int,
        bias = True,
    ):
        super().__init__()
        self.in_feat = in_feat
        self.regression = nn.Linear(in_feat, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, attentions):
        
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        #features: BxCxTxT
        attentions = attentions.to(next(self.parameters()))#attention always float32, may be needed to convert to float16
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0,2,3,1)
        return self.activation(self.regression(attentions).squeeze(3))        





###########################################################

#ROW AND COLUMNS SELF ATTENTIONS



class RowSelfAttention(nn.Module):
    """Self-attention over rows"""

    def __init__(self,
        embedding_dim,
        num_att_heads,
        dropout,
        max_tokens: int =2**13,
    ):
        super().__init__()

        #Inizializzo parametri
        self.num_att_heads = num_att_heads
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens

        #d_v dimension of keys and queries
        # // is floor division
        self.d_k = self.embedding_dim // self.num_att_heads

        self.scaling = self.d_k ** -0.5

        self.att_shape = "hnij"

        self.k_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim,self.embedding_dim)

        self.out_proj = nn.Linear(self.embedding_dim,self.embedding_dim)

        self.dropout = nn.Dropout(self.dropout)

    
    def align_scaling(self, q):
        n_rows = q.size(0)
        return self.scaling / math.sqrt(n_rows)

    def b_forward(
        self,
        x,
        padding_mask = None,
    ):
        n_rows, n_cols, batch_size, embedding_dim = x.size()

        # // is floor division
        max_rows = max(1, self.max_tokens // n_cols)

        attentions = 0
        scaling = self.align_scaling(x)
        for start in range(0, n_rows, max_rows):
            attention_weights = self.compute_attention_weights(
                x[start : start + max_rows],
                scaling,
                padding_mask[:, start : start + max_rows]
                if padding_mask is not None
                else None,
            )

            attentions += attention_weights
        attention_probabilities = attentions.softmax(-1)
        attention_probabilities = self.dropout(attention_probabilities)

        outputs = []
        for start in range(0, n_rows, max_rows):
            o = self.update_attention_weights(x[start : start + max_rows], attention_probabilities)
            outputs.append(o)

        output = torch.cat(outputs, 0)

        return output, attention_probabilities




    def compute_attention_weights(
        self,
        x,
        scaling: float,
        padding_mask = None,
    ):
        n_rows, n_cols, batch_size, embedding_dim = x.size()
        q = self.q_proj(x).view(n_rows, n_cols, batch_size, self.num_att_heads, self.d_k)
        k = self.k_proj(x).view(n_rows, n_cols, batch_size, self.num_att_heads, self.d_k)

        q *= scaling
        if padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= (1 - padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(4).to(q))

        attention_weights = torch.einsum(f"rinhd,rjnhd->hnij", q, k)

        if padding_mask is not None:
            attention_weights = attention_weights.masked_fill(
                padding_mask[:,0].unsqueeze(0).unsqueeze(2),
                -10000,
            )

        return attention_weights


    def update_attention_weights(
        self,
        x,
        attention_probabilities,
    ):
        n_rows, n_cols, batch_size, embedding_dim = x.size()
        v = self.v_proj(x).view(n_rows, n_cols, batch_size, self.num_att_heads, self.d_k)
        c = torch.einsum(f"hnij,rjnhd->rinhd", attention_probabilities, v)

        #Returns a contiguous in memory tensor containing the same data as self tensor. 
        #If self tensor is already in the specified memory format, this function returns the self tensor.
        c = c.contiguous().view(n_rows, n_cols, batch_size, embedding_dim)
        output = self.out_proj(c)
        return output


    #forward pass of the row self attention layer
    def forward(
        self,
        x,
        padding_mask = None,
    ):
        n_rows, n_cols, batch_size, embedding_dim = x.size()
        if (n_rows*n_cols > self.max_tokens) and not torch.is_grad_enabled():
            return self.b_forward(x, padding_mask)
        else:
            scaling = self.align_scaling(x)
            attention_weights = self.compute_attention_weights(x, scaling, padding_mask)
            attention_probabilities = attention_weights.softmax(-1)
            attention_probabilities = self.dropout(attention_probabilities)
            output = self.update_attention_weights(x, attention_probabilities)
            return output, attention_probabilities



class ColumnSelfAttention(nn.Module):
    """Self attention over columns"""


    def __init__(self,
        embedding_dim,
        num_att_heads,
        dropout = 0.2,
        max_tokens: int =2**13,
    ):
        super().__init__()

        #Inizializzo parametri
        self.num_att_heads = num_att_heads
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens

        #d_v dimension of keys and queries
        # // is floor division
        self.d_k = self.embedding_dim // self.num_att_heads
        #TODO: capire a cosa serve scaling ---> dovrebbe essere la normalizzazione
        self.scaling = self.d_k ** -0.5

        self.att_shape = "hnij"


        self.k_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim,self.embedding_dim)

        self.out_proj = nn.Linear(self.embedding_dim,self.embedding_dim)

        self.dropout = nn.Dropout(self.dropout)

    
    


    def b_forward(
        self,
        x,
        padding_mask = None,
    ):
        n_rows, n_cols, batch_size, embedding_dim = x.size()

        # // is floor division
        max_cols = max(1, self.max_tokens // n_rows)

        outputs = []
        attentions = []

        for start in range(0, n_cols, max_cols):
            output, attention = self(
                x[:,start : start + max_cols],
                padding_mask[:, :, start : start + max_cols]
                if padding_mask is not None
                else None,
            )

            outputs.append(output)
            attentions.append(attention)

        outputs = torch.cat(outputs, 1)
        attentions = torch.cat(attentions, 1)

        return output, attentions


    def update_attention(
        self,
        x,
        padding_mask
    ):
        n_rows, n_cols, batch_size, embedding_dim = x.size()
        if n_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with padding
            attentions_probabilities = torch.ones(
                self.num_att_heads,
                n_cols,
                batch_size,
                n_rows,
                n_rows,
                device=x.device,
                dtype=x.dtype
            )
            output = self.out_proj(self.v_proj(x))
        else:
            q = self.q_proj(x).view(n_rows, n_cols, batch_size, self.num_att_heads, self.d_k)
            k = self.k_proj(x).view(n_rows, n_cols, batch_size, self.num_att_heads, self.d_k)
            v = self.v_proj(x).view(n_rows, n_cols, batch_size, self.num_att_heads, self.d_k)
            q *= self.scaling

            attention_weights = torch.einsum("icnhd,jcnhd->hcnij", q, k)
            
            if padding_mask is not None:
                attention_weights = attention_weights.masked_fill(
                    padding_mask.permute(2,0,1).unsqueeze(0).unsqueeze(3),
                    -10000,
                )

            attentions_probabilities = attention_weights.softmax(-1)
            attentions_probabilities = self.dropout(attentions_probabilities)
            c = torch.einsum("hcnij,jcnhd->icnhd", attentions_probabilities, v)
            c = c.contiguous().view(n_rows, n_cols, batch_size, embedding_dim)
            output = self.out_proj(c)
        return output, attentions_probabilities

    def forward(
        self,
        x,
        padding_mask = None,
    ):
        n_rows, n_cols, batch_size, embedding_dim = x.size()
        # if False and num_rows * num_cols > 2 ** 14 and not torch.is_grad_enabled():
        if (n_rows * n_cols) > self.max_tokens and not torch.is_grad_enabled():
            return self.b_forward(
                x,
                padding_mask
            )
        else:
            return self.update_attention(x, padding_mask)




#LAYERNORM

try:
    from .normalization import FusedLayerNorm as _FusedLayerNorm

    class LayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)


except ImportError:
    from torch.nn import LayerNorm as LayerNorm