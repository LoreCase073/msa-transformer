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
from accuracy import accuracy
import torch.nn as nn


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

    parser.add_argument("--epochs", dest="epochs", default=5, help="number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=64, help="Batch size")
    parser.add_argument("--lr", dest="lr", default=1e-4, help="learning rate train")
    parser.add_argument("--scheduling", dest="scheduling", default=0,
                        help="1 if scheduling lr policy applied, 0 otherwise")
    parser.add_argument("--alphabet_size", dest="alphabet_size", default=21,
                        help="size of the alphabet used, excluding padding and mask characters")
    parser.add_argument("--padding_idx", dest="padding_idx", default=21,
                        help="value of token used for padding")
    parser.add_argument("--mask_idx", dest="mask_idx", default=22,
                        help="value of token used for masking")
    parser.add_argument("--max_sequences", dest="max_sequences", default=64,
                        help="max sequences subsampled from the MSAs (rows)")
    parser.add_argument("--max_positions", dest="max_positions", default=128,
                        help="max positions subsampled from the MSAs (columns)")
    parser.add_argument("--mask_prob", dest="mask_prob", default=0.2,
                        help="probability of masking a token")
    parser.add_argument("--weights_path", dest="weights_path", default=None,
                        help="path to the folder where storing the model weights")
    parser.add_argument("--saved_weights", dest="saved_weights", default=None,
                        help="path to the folder where storing the model weights to load")                    
    parser.add_argument("--dataset_path", dest="dataset_path",
                        help="path to the dataset folder (where train, test)")
    parser.add_argument("--csv_dataset", dest="csv_dataset",
                        help="path to the csv file with the names of the training data")
    parser.add_argument("--csv_val", dest="csv_val",
                        help="path to the csv file with the names of the validation data")
    parser.add_argument("--project_name", dest="project_name",
                        help="path to the dataset folder (where train, test)")
    parser.add_argument("--name_experiment", dest="name_experiment",
                        help="path to the dataset folder (where train, test)")



    parser = MSATransf.args(parser)

    args = parser.parse_args()

    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    scheduling = int(args.scheduling)
    alphabet_size = int(args.alphabet_size)
    padding_idx = int(args.padding_idx)
    mask_idx = int(args.mask_idx)
    max_sequences = int(args.max_sequences)
    max_positions = int(args.max_positions)
    mask_prob = float(args.mask_prob)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")

    hyper_parameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "scheduling": scheduling,
        "alphabet_size": alphabet_size,
        "max_sequences": max_sequences,
        "max_positions": max_positions,
        "mask_prob": mask_prob,
        "layers": args.layers,
        "embed_dim": args.embed_dim,
        "bias": args.bias,
        "ffn_embedding_dim": args.ffn_embedding_dim,
        "attention_heads": args.attention_heads,
        "dropout": args.dropout,
        "attention_dropout": args.att_dropout,
        "activation_dropout": args.act_dropout,
        "max_tokens": args.max_tokens,
    }

    experiment = Experiment(project_name=args.project_name)
    experiment.set_name(args.name_experiment)

    msat = MSATransf(args, alphabet_size, padding_idx,
                    mask_idx)

    mask_idx = None
    #Non carico modello precedente, qui da zero addestro
    #msat.load_state_dict(torch.load(args.saved_weights))
    
    experiment.log_parameters(hyper_parameters)
    experiment.set_model_graph(msat)

    save_path = os.path.join(args.weights_path, args.name_experiment)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"Save weights in: {save_path}.")

    # save hyperparams dictionary in save_weights_path
    with open(save_path + '/hyperparams.json', "w") as outfile:
        json.dump(hyper_parameters, outfile, indent=4)
    
    #Dataset loading
    msa_dataset = MSADataset(file_csv=args.csv_dataset,
    npz=args.dataset_path,
    mask_prob=mask_prob, 
    max_seq_len=max_sequences, 
    max_pos=max_positions,
    padding_idx=padding_idx,
    mask_idx=mask_idx,
    )


    valid_set = MSADataset(file_csv=args.csv_val,
    npz=args.dataset_path,
    mask_prob=mask_prob, 
    max_seq_len=max_sequences, 
    max_pos=max_positions,
    padding_idx=padding_idx,
    mask_idx=mask_idx,
    )


    #TODO: fare evaluation loader --- dataset e dataloader

    training_loader = DataLoader(dataset=msa_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2, 
    collate_fn=collate_tensors)

    val_loader = DataLoader(dataset=valid_set, 
    batch_size=1, 
    shuffle=True, 
    num_workers=2, 
    collate_fn=collate_tensors)

    optimizer = optim.AdamW(msat.parameters(), lr)

    msat = msat.to(device)

    params = sum(p.numel() for p in msat.parameters() if p.requires_grad)

    print(f"Number of parameters in the model: {params}")
    #for name, param in msat.named_parameters():
    #    print(f"Name: {name}    Number of param: {param.shape}  Grad: {param.requires_grad}")

    experiment.log_other('n_param', params)


    print("Starting the training")

    
    
    lossFunc = nn.BCELoss() 

    for epoch in range(epochs):
        msat.train()

        running_loss = 0.0
        train_acc = 0.0
        with tqdm(training_loader, unit="batch") as tepoch:

            for msa, _, _, distances in tepoch:
                tepoch.set_description(f"Epoch{epoch}")
                inputs = msa.to(device)

                labels = distances.float().to(device)
                
                labels = padding_contacts(labels)

                
                
                

                optimizer.zero_grad()

                output = msat(inputs, return_contacts=True)
                #Not needed at this stage
                #logits = output['logits']
                #repr = output['representations']
                contacts = output['contacts'].float().to(device)
                
                contacts = padding_contacts(contacts)
                
                loss = lossFunc(contacts, labels)
                loss.backward()
                optimizer.step()
                contacts = torch.squeeze(contacts)
                labels = torch.squeeze(labels)
                acc = accuracy(contacts, labels)

                

                train_acc += acc['LR_L/5']
                running_loss += loss.item()
            time.sleep(0.1)

        
        
        print(f"Training loss: {running_loss/len(training_loader)} Epoch: {epoch}")
        experiment.log_metric('train_loss', running_loss/len(training_loader), step=epoch+1)
        experiment.log_metric('train_accuracy_LR_L/5', train_acc/len(training_loader), step=epoch+1)
        if epoch % 1 ==0:
            torch.save(msat.state_dict(), save_path + '/weights_' + (str(epoch+1)) + '.pth')

        #START of the evaluation!
        if epoch % 5 == 0:
            print(f'START EVALUATION:')
            msat.eval() 
            val_loss = 0.0
            lrl5 = 0.0
            lrl2 = 0.0
            lrl = 0.0
            with torch.no_grad():
                with tqdm(val_loader, unit='batch') as vepoch:
                    for msa, _, _, distances in vepoch:
                        tepoch.set_description(f"Epoch{epoch}")
                        inputs = msa.to(device)

                        labels = distances.float().to(device)

                        labels = padding_contacts(labels)

                        optimizer.zero_grad()

                        output = msat(inputs, return_contacts=True)

                        contacts = output['contacts'].float().to(device)

                        contacts = padding_contacts(contacts)
                        
                        loss = lossFunc(contacts, labels)
                        contacts = torch.squeeze(contacts)
                        labels = torch.squeeze(labels)
                        vacc = accuracy(contacts, labels)
                        lrl5 += vacc['LR_L/5']
                        lrl2 += vacc['LR_L/2']
                        lrl += vacc['LR_L']
                        val_loss += loss.item()
                    time.sleep(0.1)
                print('Evaluation Loss: ' + str(val_loss/len(val_loader)))
                experiment.log_metric('VAL_LOSS: ', val_loss/len(val_loader), step = epoch + 1)
                experiment.log_metric('VAL_ACCURACY_LR_L/5: ', lrl5/len(val_loader), step = epoch + 1)
                experiment.log_metric('VAL_ACCURACY_LR_L/2: ', lrl2/len(val_loader), step = epoch + 1)
                experiment.log_metric('VAL_ACCURACY_LR_L: ', lrl/len(val_loader), step = epoch + 1)
                print(f'END EVALUATION {epoch}:')
        
    
    torch.save(msat.state_dict(), save_path + '/weights_' + 'final.pth')

    experiment.end()
    print('End of the training')