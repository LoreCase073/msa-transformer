import math
from turtle import forward
from typing import Optional
from sympy import symmetrize
import torch
from torch import embedding
from torch import narrow
import torch.nn as nn
import torch.nn.functional as F
from attention import AxialTL, LayerNorm, ContPredictionHead, RowSelfAttention,ColumnSelfAttention, LMHead
from embedding_pos import PositionalEmbedding


class MSATransf(nn.Module):
    """MSA transformer inspired model"""
    #define the arguments of the model
    @classmethod
    def args(c, parser):
        
        parser.add_argument(
            "--layers",
            default=4,
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
            default=6,
            type=int,
            metavar="N",
            help=" # of attentions heads"
        )

        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            help="Dropout"
        )

        parser.add_argument(
            "--att_dropout",
            default=0.1,
            type=float,
            help="Dropout to apply on the attention"
        )

        parser.add_argument(
            "--act_dropout",
            default=0.1,
            type=float,
            help="Dropout to apply on the activation"
        )

        parser.add_argument(
            "--max_tokens",
            default=2**14,
            type=int,
            help=(
                "Used during inference to batch attention computations in a single "
                "forward pass. This allows increased input sizes with less memory."
            )
        )

        return parser

    #initialize the model
    def __init__(self,args,alphabet_size, padding_idx, mask_idx):
        super().__init__()
        self.args = args,
        self.args = self.args[0]
        #gli idx sono tutti indici legati ai tokens
        self.alphabet_size = alphabet_size + 1 + 1#+1 finale è per il padding idx
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.cls_idx = None
        self.eos_idx = None
        self.prepend_bos = None
        self.append_eos = None


        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )


        #TODO: rimuovere parte di positional embedding in colonna
        '''
        if getattr(self.args, "embed_msa_pos", False):
            embedding_dim = getattr(self.args, "embed_msa_pos_dim", self.args.embed_dim)
            self.msa_position_embedding = nn.parameter.Parameter(
                0.01 * torch.randn(1, 1024, 1, embedding_dim),
                requires_grad=True,
            )
        else:
            self.register_parameter("embed_msa_pos", None)

        '''

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
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

        self.embed_pos = PositionalEmbedding(
            #TODO: max_pos non in args, capire da dove deriva o come inserirlo
            int(self.args.max_positions),
            int(self.args.embed_dim),
            self.padding_idx,
        )

        self.emb_layer_norm_b = LayerNorm(self.args.embed_dim)
        self.emb_layer_norm_a = LayerNorm(self.args.embed_dim)

        self.lm_head = LMHead(
            embedding_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight = self.embed_tokens.weight,
        )



    #forward pass
    def forward(self, tokens, repr_layers = [], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim ==3
        batch_size, num_align, seqlen = tokens.size()
        #TODO: capire a cosa serve questa linea, token.eq? cosa sono B,R,C?
        #RISPOSTA: calcola element-wise equality ---> sto creando la maschera dell'input ---> vede dove è segnato il padding, ovvero elemento mancante
        #B = batch, R = row, C = column
        padding_mask = tokens.eq(self.padding_idx) #B,R,C
        mask = tokens.eq(self.mask_idx)
        if not padding_mask.any():
            padding_mask = None
        if not mask.any():
            mask = None
        
        x = self.embed_tokens(tokens)
        #TODO: cosa fa questa riga sotto?
        x += self.embed_pos(tokens.view(batch_size*num_align, seqlen)).view(x.size())






        #TODO: rimuovere questa, dovrebbe essere embedding delle righe per posizione
        '''
        if self.msa_position_embedding is not None:
            if x.size(1) > 1024:
                raise RuntimeError(
                    "Using model with MSA position embedding trained on maximum MSA "
                    f"depth of 1024, but received {x.size(1)} alignments."
                )
            x +=  self.msa_position_embedding[:, :num_align]
        '''




        
        
        x = self.emb_layer_norm_b(x)

        x = self.dropout_mod(x)

        if padding_mask is not None:
            #TODO: cosa fa qui? mi sa l'idea che stia tipo eliminando elementi da maskerare
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))


        #TODO: cosa è questo?
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
                mask=mask,
                padding_mask = padding_mask,
                need_weights = need_head_weights,
            )
            #TODO: perchè si fanno queste permutazioni?
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

        #last hidden repr should have a layer norm

        if(layer_idx+1) in repr_layers:
            hidden_repr[layer_idx+1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_repr}
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


        

    #compute contact prediction
    def predict(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    #used to get the number of layers
    def n_layers(self):
        return self.args.layers


    #define the max number of tokens of an MSA
    def max_tokens(self, value: int) -> None:
        for module in self.modules():
            if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
                module.max_tokens = value







###########################################################




###########################################################

#