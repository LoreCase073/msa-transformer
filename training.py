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



if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Train MSAT')

    parser.add_argument("--epochs", dest="epochs", default=5, help="number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=64, help="Batch size")
    parser.add_argument("--lr", dest="lr", default=1e-4, help="learning rate train")
    parser.add_argument("--scheduling", dest="scheduling", default=0,
                        help="1 if scheduling lr policy applied, 0 otherwise")
    parser.add_argument("--alphabet_size", dest="alphabet_size", default=21,
                        help="size of the alphabet used, excluding padding and mask characters")
    parser.add_argument("--padding_idx", dest="padding_idx", default=22,
                        help="value of token used for padding")
    parser.add_argument("--mask_idx", dest="mask_idx", default=21,
                        help="value of token used for masking")
    parser.add_argument("--max_sequences", dest="max_sequences", default=64,
                        help="max sequences subsampled from the MSAs (rows)")
    parser.add_argument("--max_positions", dest="max_positions", default=128,
                        help="max positions subsampled from the MSAs (columns)")
    parser.add_argument("--mask_prob", dest="mask_prob", default=0.2,
                        help="probability of masking a token")
    parser.add_argument("--weights_path", dest="weights_path", default=None,
                        help="path to the folder where storing the model weights")
    parser.add_argument("--dataset_path", dest="dataset_path",
                        help="path to the dataset folder (where train, test)")
    parser.add_argument("--csv_dataset", dest="csv_dataset",
                        help="path to the csv file with the names of the training data")
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

    training_loader = DataLoader(dataset=msa_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2, 
    collate_fn=collate_tensors)

    optimizer = optim.AdamW(msat.parameters(), lr)

    msat = msat.to(device)

    model_param = filter(lambda p: p.requires_grad, msat.parameters())
    params = sum([np.prod(p.size()) for p in model_param])

    print(f"Number of parameters in the model: {params}")

    experiment.log_other('n_param', params)


    print("Starting the training")
    ignore_idx = -100

    for epoch in range(epochs):
        msat.train()

        running_loss = 0.0
        train_acc = 0.0
        train_acc_msa = 0.0
        with tqdm(training_loader, unit="batch") as tepoch:

            for msa, masked, mask, distances in tepoch:
                tepoch.set_description(f"Epoch{epoch}")
                inputs = masked.to(device)

                labels = msa.masked_fill(~mask, ignore_idx).long().to(device)
                print(labels)

                optimizer.zero_grad()

                output = msat(inputs)
                logits = output['logits']
                
                loss = F.cross_entropy(
                    logits.permute(0,3,1,2),
                    labels,
                    ignore_index=ignore_idx
                )
                loss.backward()
                optimizer.step()

                acc = ((logits.argmax(dim=-1) == labels)&(labels != padding_idx)).float().sum()
                num_labels = (labels != ignore_idx).float().sum()
                acc = acc / num_labels

                #TODO: mettere di ignorare padding? ---> acc = ((logits.argmax(dim=-1) == labels)&(labels != padding_idx)).float().sum()
                acc_msa = ((logits.argmax(dim=-1) == msa.to(device))&(msa.to(device)!=padding_idx)).float().sum()
                acc_msa = acc_msa / (msa.shape[0]*msa.shape[1]*msa.shape[2])

                train_acc += acc
                running_loss += loss.item()
                train_acc_msa += acc_msa
            time.sleep(0.1)
        
        
        print(f"Training loss: {running_loss/len(training_loader)} Epoch: {epoch}")
        experiment.log_metric('train_loss', running_loss/len(training_loader), step=epoch+1)
        experiment.log_metric('train_accuracy', train_acc/len(training_loader), step=epoch+1)
        experiment.log_metric('train_accuracy_on_msa', train_acc_msa/len(training_loader), step=epoch+1)
        if epoch % 20 ==0:
            torch.save(msat.state_dict(), save_path + '/weights_' + (str(epoch+1)) + '.pth')
    
    torch.save(msat.state_dict(), save_path + '/weights_' + 'final.pth')

    experiment.end()
    print('End of the training')