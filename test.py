from comet_ml import Experiment
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os
from model import MSATransf
from tqdm import tqdm
import time
import argparse
from dataset import MSADataset, collate_tensors
import json
import numpy as np
from precision import precision
import torch.nn as nn
import os
import csv

from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable

def padding_contacts(x):
    batch_size, seqlen, _ = x.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = (sep >= 6) | (sep<=-6)

    x = x.masked_fill(~valid_mask, float(0))
    
    
    return x



if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Train MSAT - Contacts')

    parser.add_argument("--batch_size", dest="batch_size", default=1, help="Batch size")
    parser.add_argument("--alphabet_size", dest="alphabet_size", default=21,
                        help="size of the alphabet used, excluding padding")
    parser.add_argument("--padding_idx", dest="padding_idx", default=21,
                        help="value of token used for padding")
    parser.add_argument("--max_sequences", dest="max_sequences", default=64,
                        help="max sequences subsampled from the MSAs (rows)")
    parser.add_argument("--max_positions", dest="max_positions", default=128,
                        help="max positions subsampled from the MSAs (columns)")
    parser.add_argument("--saved_weights", dest="saved_weights", default=None,
                        help="path to the folder where storing the model weights to load")                    
    parser.add_argument("--dataset_path", dest="dataset_path",
                        help="path to the dataset folder (where train, test)")
    parser.add_argument("--csv_test", dest="csv_test",
                        help="path to the csv file with the names of the test data")
    parser.add_argument("--project_name", dest="project_name",
                        help="path to the dataset folder (where train, test)")
    parser.add_argument("--name_experiment", dest="name_experiment",
                        help="path to the dataset folder (where train, test)")
    parser.add_argument("--base_arch", dest="base_arch",
                        help="Name of the pretraining architecture used")


    parser = MSATransf.args(parser)

    args = parser.parse_args()

    batch_size = int(args.batch_size)
    alphabet_size = int(args.alphabet_size)
    padding_idx = int(args.padding_idx)
    max_sequences = int(args.max_sequences)
    max_positions = int(args.max_positions)


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)

    hyper_parameters = {
        "batch_size": batch_size,
        "alphabet_size": alphabet_size,
        "max_sequences": max_sequences,
        "max_positions": max_positions,
        "layers": args.layers,
        "embed_dim": args.embed_dim,
        "bias": args.bias,
        "ffn_embedding_dim": args.ffn_embedding_dim,
        "attention_heads": args.attention_heads,
        "dropout": args.dropout,
        "attention_dropout": args.att_dropout,
        "activation_dropout": args.act_dropout,
        "max_tokens": args.max_tokens,
        "base_architecture": args.base_arch,
    }

    experiment = Experiment(project_name=args.project_name)
    experiment.set_name(args.name_experiment)

    msat = MSATransf(args, alphabet_size, padding_idx)

    #Carico modello precedente
    msat.load_state_dict(torch.load(args.saved_weights))
    
    experiment.log_parameters(hyper_parameters)
    experiment.set_model_graph(msat)



    
    #Dataset loading

    test_set = MSADataset(file_csv=args.csv_test,
    npz=args.dataset_path,
    max_seq_len=max_sequences, 
    max_pos=max_positions,
    padding_idx=padding_idx,
    )



    test_loader = DataLoader(dataset=test_set, 
    batch_size=1, 
    shuffle=True, 
    num_workers=2, 
    collate_fn=collate_tensors)


    msat = msat.to(device)

    model_param = filter(lambda p: p.requires_grad, msat.parameters())
    params = sum([np.prod(p.size()) for p in model_param])

    print(f"Number of parameters in the model: {params}")

    experiment.log_other('n_param', params)


    #START of the test!

    print(f'START TEST:')
    msat.eval()
    lrl5 = 0.0
    lrl2 = 0.0
    lrl = 0.0


    csv_list = []

    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as vepoch:
            for msa, distances,name in vepoch:
                inputs = msa.to(device)

                labels = distances.float().to(device)

                labels = padding_contacts(labels)

                output = msat(inputs, return_contacts=True)
                
                

                contacts = output['contacts'].float().to(device)

                contacts = padding_contacts(contacts)
                
                contacts = torch.squeeze(contacts)
                labels = torch.squeeze(labels)
                v_met = precision(contacts, labels)
                csv_list.append((name, str(v_met['LR_L']), str(v_met['LR_L/5'])))
                lrl5 += v_met['LR_L/5']
                lrl2 += v_met['LR_L/2']
                lrl += v_met['LR_L']
            time.sleep(0.1)
    experiment.log_metric('Test_Precision_LR_L/5: ', lrl5/len(test_loader))
    experiment.log_metric('Test_Precision_LR_L/2: ', lrl2/len(test_loader))
    experiment.log_metric('Test_Precision_LR_L: ', lrl/len(test_loader))
    print(f'END TEST')
    

        

    experiment.end()
    print('End of the training')