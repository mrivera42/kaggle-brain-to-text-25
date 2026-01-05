import h5py
import pandas as pd
import os 
import torch 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 
import numpy as np
import torch.nn.functional as F
import torch.optim as optim 

import torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist 

import time 
import json

from dataset import NeuralDataset
from rnn import BaselineLSTM 
import matplotlib.pyplot as plt 


# each gpu runs one process 
# setting up a group is necessary so that gpus can discover and communicate with eachother 
# world size = total number of processes in a group 
# rank = unique id for each process 
# ddp will launch a process on each GPU 
# each process is going to initalize the trainer function, with a copy of the model and optimizer 

# only compute metrics and only save checkpoints from the rank zero process 
# since the models on each gpu are the same, we want to avoid redundancy 

# distributed sampler ensures that the input batch is chunked across all gpus without overlap 

# ddp = distributed data parallel - essentially the full model weights are copied onto each GPU, and the data is split across GPUs 


def ddp_setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():

    dist.destroy_process_group()



LOGIT_TO_PHONEME = [
   # "BLANK" = CTC blank symbol
'AA', 'AE', 'AH', 'AO', 'AW',
'AY', 'B', 'CH', 'D', 'DH',
'EH', 'ER', 'EY', 'F', 'G',
'HH', 'IH', 'IY', 'JH', 'K',
'L', 'M', 'N', 'NG', 'OW',
'OY', 'P', 'R', 'S', 'SH',
'T', 'TH', 'UH', 'UW', 'V',
'W', 'Y', 'Z', 'ZH',
' | ',    # "|" = silence token
] 


def load_h5py_file(file_path):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
    return data









# create dataloader 


def collate_fn(batch):

    # gather all the variable length items (neural_features, seq_labels) from the batch into separate python lists first so we can find the max length of each group and pad each group together

    neural_features = [i['neural_features'] for i in batch]
    seq_class_ids = [i['seq_class_ids'] for i in batch]
    n_time_steps = [i['n_time_steps'] for i in batch]
    seq_len = [i['seq_len'] for i in batch] 
    transcription = [i['transcription'] for i in batch]
    sentence_label = [i['sentence_label'] for i in batch]
    session = [i['session'] for i in batch]
    block_num = [i['block_num'] for i in batch]
    trial_num = [i['trial_num'] for i in batch]

    neural_features_padded = pad_sequence(neural_features, batch_first=True, padding_value=0)
    seq_class_ids_padded = pad_sequence(seq_class_ids, batch_first=True, padding_value=0)

    return {
        'neural_features': neural_features_padded,
        'seq_class_ids': seq_class_ids_padded,
        'n_time_steps': n_time_steps,
        'seq_len': seq_len,
        'transcription': transcription,
        'sentence_label': sentence_label,
        'session': session,
        'block_num': block_num,
        'trial_num': trial_num
    }

 

# define training loop 
def train(model, trainloader, valloader, rank):

    print(f'starting training loop on device {rank}')
    start_time = time.time()

    metadata = {}
    lr = 1e-5
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    loss_fn = torch.nn.CTCLoss()
    num_epochs = 2
    patience = 3

    model.train()

    epoch_loss = 0
    train_losses = []
    val_losses = []
    early_stopping_counter = 0
    best_val = -100
    for epoch in range(num_epochs):

        
        train_loss = 0
        num_batches = len(trainloader)
        num_val_batches = len(valloader)
        for i, batch in enumerate(trainloader):

            # place tensors on device
            inputs, targets = batch['neural_features'].to(rank), batch['seq_class_ids'].to(rank)
            inputs = torch.transpose(inputs, 0, 1)
            # print(f'inputs shape: {inputs.shape}')
            # print(f'targets shape: {targets.shape}')
            input_lengths, target_lengths = torch.tensor(batch['n_time_steps']).to(rank), torch.tensor(batch['seq_len']).to(rank)
            # print(f'input_lengths shape: {input_lengths.shape}')
            # print(f'target_lengths shape: {target_lengths.shape}')

            # zero optimizer
            optimizer.zero_grad()

            # forward 
            output = model.forward(inputs)
            # print(f'output shape: {output.shape}' )

            # compute loss 
            loss = loss_fn(output, targets, input_lengths, target_lengths)
            train_loss += loss.item()
            

            # backprop 
            loss.backward()

            # update weights 
            optimizer.step()
        
        train_loss /= num_batches 
        train_losses.append(train_loss)

        # switch to eval mode 
        if rank == 0:
            model.eval()
            val_loss = 0 
            for i, batch in enumerate(valloader): 

                # put data onto device 
                inputs, targets = batch['neural_features'].to(rank), batch['seq_class_ids'].to(rank)
                inputs = torch.transpose(inputs, 0, 1) # ctc loss expects this for some reason

                # forward 
                with torch.no_grad():

                    output = model(inputs)

                # compute loss 
                input_lengths, target_lengths = torch.tensor(batch['n_time_steps']).to(rank), torch.tensor(batch['seq_len']).to(rank)
                loss = loss_fn(output, targets, input_lengths, target_lengths)

                val_loss += loss.item()

            val_loss /= num_val_batches
            val_losses.append(val_loss)
            curr_val = val_loss

            # early stopping 
            if curr_val < best_val:
                best_val = curr_val 
            
            if curr_val > best_val:
                early_stopping_counter += 1 
            
            if early_stopping_counter > patience and rank == 0:

                print(f'early stopping initiated: early_stopping_counter: {early_stopping_counter}, patience: {patience}')

            # save model 
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            cpu_state = {k: v.cpu() for k,v in state_dict.items()}
            torch.save({'model_state_dict': cpu_state, 'metadata': metadata}, 'checkpoint.pt')

            # save metadata 
            with open('train_metadata.json', 'w') as f: 
                json.dump(metadata, f, indent=2)


            # print out stats for this epoch 
            end_time = time.time()
            epoch_duration = (end_time - start_time) / 60
            print(f'Epoch {epoch} train loss: {train_loss:.2f}, val loss: {val_loss:.2f}, took: {epoch_duration:.2f} minutes')

            # save training plot 
            train_losses.append(train_loss)
            
            # stop training 
            return model, metadata


        if rank == 0:
            end_time = time.time()
            epoch_duration = (end_time - start_time) / 60
            print(f'Epoch {epoch} train loss: {train_loss:.2f}, val loss: {val_loss:.2f}, took: {epoch_duration:.2f} minutes')

    metadata['num_epochs'] = num_epochs
    metadata['lr'] = lr 
    metadata['train_losses'] = train_losses

    if rank == 0: 

        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('epoch')
        plt.ylabel('ctc loss')
        plt.legend()
        plt.savefig('losses.png')
        plt.close()

    return model, metadata



def worker(rank, world_size):

    torch.cuda.set_device(rank)

    ddp_setup(rank, world_size)

    filepath = 't15_copyTask_neuralData/hdf5_data_final'

    print('about to load datasets')

    trainDataset = NeuralDataset(filepath)
    print('loaded train set')
    valDataset = NeuralDataset(filepath,val=True)
    print('loaded val set')

    print('about to create dataloaders')
    trainSampler = DistributedSampler(dataset=trainDataset, num_replicas=world_size, rank=rank, shuffle=True)
    valSampler = DistributedSampler(dataset=valDataset,num_replicas=world_size, rank=rank, shuffle=True)
    trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=False, collate_fn=collate_fn, sampler=trainSampler)
    print('finished trainloader')
    valLoader = DataLoader(valDataset,batch_size=1, shuffle=False, collate_fn=collate_fn, sampler=valSampler)
    print('finished val loader')

    model = BaselineLSTM().to(rank)
    model = DDP(model,device_ids=[rank])
    print('initialized model')

    model, metadata = train(model, trainLoader, valLoader, rank)
    
    if rank == 0: 

        # save model 
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        cpu_state = {k: v.cpu() for k,v in state_dict.items()}
        torch.save({'model_state_dict': cpu_state, 'metadata': metadata}, 'checkpoint.pt')

        # save metadata 
        with open('train_metadata.json', 'w') as f: 
            json.dump(metadata, f, indent=2)
        

    # print(metadata)
    cleanup()

    





if __name__ == "__main__": 

    
    world_size = 2
    start_time = time.time()
    try:
        mp.spawn(worker, nprocs=world_size, args=(world_size,))
    except KeyboardInterrupt:
        cleanup()
        torch.cuda.empty_cache()
    end_time = time.time()
    duration = (end_time - start_time) / 60 
    print(f'Total training time: {duration:.2f} minutes')