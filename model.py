import math
from turtle import forward
from typing import Optional
from sympy import symmetrize
import torch
from torch import embedding
from torch import narrow
import torch.nn as nn
import torch.nn.functional as F
from attention_architecture import AxialTL, LayerNorm, ContPredictionHead, RowSelfAttention,ColumnSelfAttention
from embedding_pos import PositionalEmbedding


class MSATransf(nn.Module):
    """MSA transformer inspired model"""
    #define the arguments of the model
    @classmethod
    def args(c, parser):
        
        parser.add_argument(
            "--layers",
            default=12,
            type=int,
            metavar="N",
            help="number of layers in the model"
        )

        parser.add_argument(
            "--embed_dim",
            default=384,
            type=int,
            metavar="N",
            help="Embedding dimension"
        )

        parser.add_argument(
            "--bias",
            action="store_true",
            help="If apply bias to logits is what you want"
        )

        parser.add_argument(
            "--ffn_embedding_dim",
            default=768,
            type=int,
            metavar="N",
            help="embedding dimension for the fully connected net"
        )

        parser.add_argument(
            "--attention_heads",
            default=12,
            type=int,
            metavar="N",
            help=" # of attentions heads"
        )

        parser.add_argument(
            "--dropout",
            default=0.2,
            type=float,
            help="Dropout"
        )

        parser.add_argument(
            "--att_dropout",
            default=0.2,
            type=float,
            help="Dropout to apply on the attention"
        )

        parser.add_argument(
            "--act_dropout",
            default=0.2,
            type=float,
            help="Dropout to apply on the activation"
        )

        parser.add_argument(
            "--max_tokens",
            default=2**13,
            type=int,
            help=()
        )

        return parser

    #initialize the model
    def __init__(self,args,alphabet_size, padding_idx):
        super().__init__()
        self.args = args,
        self.args = self.args[0]
        self.alphabet_size = alphabet_size + 1
        self.padding_idx = padding_idx
        


        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )


        self.msa_position_embedding = nn.parameter.Parameter(
            0.01 * torch.randn(1, 1024, 1, self.args.embed_dim),
            requires_grad=True,
        )


        self.dropout_mod = nn.Dropout(self.args.dropout)
        self.layers = nn.ModuleList(
            [AxialTL(
                self.args.embed_dim,
                self.args.embed_dim,
                self.args.attention_heads,
                self.args.dropout,
                self.args.att_dropout,
                self.args.act_dropout,
                getattr(self.args, "max_tokens", self.args.max_tokens)
            )
            for _ in range(self.args.layers)
            ]
        )

        self.contact_h = ContPredictionHead(
            int(self.args.layers) * int(self.args.attention_heads),
        )

        self.embed_pos = PositionalEmbedding(
            int(self.args.max_positions),
            int(self.args.embed_dim),
            self.padding_idx,
        )

        self.emb_layer_norm_b = LayerNorm(self.args.embed_dim)
        self.emb_layer_norm_a = LayerNorm(self.args.embed_dim)





    #forward pass
    def forward(self, tokens, repr_layers = [], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True
            

        assert tokens.ndim ==3
        batch_size, num_align, seqlen = tokens.size()
        #B = batch, R = row, C = column
        padding_mask = tokens.eq(self.padding_idx) #B,R,C
        if not padding_mask.any():
            padding_mask = None
        
        x = self.embed_tokens(tokens)
        
        x += self.embed_pos(tokens.view(batch_size*num_align, seqlen)).view(x.size())

        x +=  self.msa_position_embedding[:, :num_align]
       
        
        x = self.emb_layer_norm_b(x)

        x = self.dropout_mod(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_repr = {}
        if 0 in repr_layers:
            hidden_repr[0] = x

        if need_head_weights:
            row_att_weights = []
            col_att_weights = []

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                padding_mask = padding_mask,
                need_weights = need_head_weights,
            )
            if need_head_weights:
                x, col_att, row_att = x
                # H x C x B x R x R -> B x H x C x R x R
                col_att_weights.append(col_att.permute(2, 0, 1, 3, 4))
                # H x B x C x C -> B x H x C x C
                row_att_weights.append(row_att.permute(1, 0, 2, 3))

            if (layer_idx+1) in repr_layers:
                hidden_repr[layer_idx+1] = x.permute(2,0,1,3)

        x = self.emb_layer_norm_a(x)
        x = x.permute(2,0,1,3) # R x C x B x D -> B x R x C x D


        if(layer_idx+1) in repr_layers:
            hidden_repr[layer_idx+1] = x
        last_rep = x

        result = {"logits": x, "representations": last_rep}
        if need_head_weights:
            # col_attentions: B x L x H x C x R x R
            col_attentions = torch.stack(col_att_weights, 1)
            # row_attentions: B x L x H x C x C
            row_attentions = torch.stack(row_att_weights, 1)

            result["col_attentions"] = col_attentions
            result["row_attentions"] = row_attentions
            if return_contacts:
                contacts = self.contact_h(tokens, row_attentions)
                result["contacts"] = contacts

        return result
