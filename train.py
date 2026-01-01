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





class NeuralDataset(torch.utils.data.Dataset): 

    def __init__(self, dir):
        
        self.data = {
            'neural_features': [],
            'n_time_steps': [],
            'seq_class_ids': [], 
            'seq_len': [], 
            'transcription': [], 
            'sentence_label': [], 
            'session': [], 
            'block_num': [], 
            'trial_num': []
        }

        for folder, __, files in os.walk(dir): 

            if 'data_train.hdf5' in files: 

                # load file 
                f = h5py.File(os.path.join(folder, 'data_train.hdf5'))

                # loop through trials 
                for i in list(f.keys()): 

                    trial = f[i]

                    neural_features = trial['input_features'][:]
                    n_time_steps = trial.attrs['n_time_steps']
                    seq_class_ids = trial['seq_class_ids'][:] if 'seq_class_ids' in trial else None
                    seq_len = trial.attrs['seq_len'] if 'seq_len' in trial.attrs else None
                    transcription = trial['transcription'][:] if 'transcription' in trial else None
                    sentence_label = trial.attrs['sentence_label'][:] if 'sentence_label' in trial.attrs else None
                    session = trial.attrs['session']
                    block_num = trial.attrs['block_num']
                    trial_num = trial.attrs['trial_num']

                    # append trial features to data list 
                    self.data['neural_features'].append(neural_features)
                    self.data['n_time_steps'].append(n_time_steps)
                    self.data['seq_class_ids'].append(seq_class_ids)
                    self.data['seq_len'].append(seq_len)
                    self.data['transcription'].append(transcription)
                    self.data['sentence_label'].append(sentence_label)
                    self.data['session'].append(session)
                    self.data['block_num'].append(block_num)
                    self.data['trial_num'].append(trial_num)

    def __len__(self): 

        return len(self.data['neural_features'])

    def __getitem__(self, idx): 

        return {
            'neural_features': torch.tensor(self.data['neural_features'][idx]),
            'n_time_steps': torch.tensor(self.data['n_time_steps'][idx]),
            'seq_class_ids': torch.tensor(self.data['seq_class_ids'][idx]),
            'seq_len': torch.tensor(self.data['seq_len'][idx]),
            'transcription': self.data['transcription'][idx],
            'sentence_label': self.data['sentence_label'][idx],
            'session': self.data['session'][idx],
            'block_num': self.data['block_num'][idx],
            'trial_num': self.data['trial_num'][idx]
        }



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
    # neural_lengths = [len(i) for i in neural_features]
    # seq_class_lengths = [len(i) for i in seq_class_ids]

    # max_neural_idx = np.argmax(neural_lengths)
    # max_seq_class_idx = np.argmax(seq_class_lengths)

    # max_neural_len = neural_lengths[max_neural_idx]
    # max_seq_len = seq_class_lengths[max_seq_class_idx]

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

class BaselineLSTM(torch.nn.Module):

    def __init__(self):

        super().__init__()

        # input (B x T x 512) --> output (B x T x 41)

        self.rnn = torch.nn.LSTM(input_size=512,hidden_size=768, num_layers=5)
        self.proj = torch.nn.Linear(in_features=768, out_features=41)


    def forward(self, x): 
        # print('rnn input: ', x.shape)
        x, _ = self.rnn(x)
        # print('linear input: ', x.shape)
        x = F.log_softmax(self.proj(x),dim=2)

        return x 

# define training loop 
def train(model, dataloader, rank):

    print(f'starting training loop on device {rank}')
    start_time = time.time()

    metadata = {}
    lr = 1e-5
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    loss_fn = torch.nn.CTCLoss()
    num_epochs = 100

    model.train()

    epoch_loss = 0
    train_losses = []
    for epoch in range(num_epochs):

        
        train_loss = 0
        num_batches = len(dataloader)
        for i, batch in enumerate(dataloader):

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
            train_losses.append(train_loss)

            # backprop 
            loss.backward()

            # update weights 
            optimizer.step()
        
        train_loss /= num_batches 

        if rank == 0:
            end_time = time.time()
            epoch_duration = (end_time - start_time) / 60
            print(f'Epoch {epoch} loss: {train_loss:.2f}, took: {epoch_duration:.2f} minutes')

    metadata['num_epochs'] = num_epochs
    metadata['lr'] = lr 
    metadata['train_losses'] = train_losses

    return model, metadata



def worker(rank, world_size):

    torch.cuda.set_device(rank)

    ddp_setup(rank, world_size)

    filepath = 't15_copyTask_neuralData/hdf5_data_final'

    brainDataset = NeuralDataset(filepath)

    sampler = DistributedSampler(dataset=brainDataset, num_replicas=world_size, rank=rank, shuffle=True)

    trainLoader = DataLoader(brainDataset, batch_size=32, shuffle=False, collate_fn=collate_fn, sampler=sampler)
    
    model = BaselineLSTM().to(rank)
    model = DDP(model,device_ids=[rank])

    model, metadata = train(model, trainLoader, rank)
    
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
    mp.spawn(worker, nprocs=world_size, args=(world_size,))
    end_time = time.time()
    duration = (end_time - start_time) / 60 
    print(f'Total training time: {duration:.2f} minutes')