python training.py --epochs 100 --batch_size 1 --lr 1e-4 --alphabet_size 21 --padding_idx 22 --mask_idx 21 --max_sequences 32 --max_positions 256 --mask_prob 0.2 --weights_path ./model_weights  --dataset_path ./Data/training_set_Rosetta/training_set/npz --csv_dataset ./training_set_pretrain.csv --project_name MSAT --name_experiment attempt1 --layers 12 --embed_dim 384 --ffn_embedding_dim 768 --attention_heads 6 --dropout 0.1 --att_dropout 0.1 --act_dropout 0.1 --count_par True