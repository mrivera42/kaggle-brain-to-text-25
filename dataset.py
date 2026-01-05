import torch 
import os 
import h5py


class NeuralDataset(torch.utils.data.Dataset): 

    def __init__(self, dir, val: bool = False):
        super().__init__()

        # create a list of the filepaths that we can index 
        self.index = []

        filename = 'data_val.hdf5' if val else 'data_train.hdf5'

        for folder, __, files in os.walk(dir):

            if filename in files: 

                path = os.join(folder, filename) 
                f = h5py.File(path)

                for i in list(f.keys()):       


                    entry = {'filepath': path, 'trial': i}
                    self.index.append(entry)

    
    def __len__(self): 

        return len(self.index)
    
    def __get_item__(self, idx): 

        file_path, trial_name = self.index[idx]['filepath'], self.index[idx]['trial']

        f = h5py.File(file_path)
        trial = f[trial_name]

        item = {
            'neural_features': torch.tensor(trial['input_features'][:]),
            'n_time_steps': torch.tensor(trial.attrs['n_time_steps']),
            'seq_class_ids': torch.tensor(trial['seq_class_ids'][:] if 'seq_class_ids' in trial else None), 
            'seq_len': torch.tensor(trial.attrs['seq_len'] if 'seq_len' in trial.attrs else None), 
            'transcription': trial['transcription'][:] if 'transcription' in trial else None, 
            'sentence_label': trial.attrs['sentence_label'][:] if 'sentence_label' in trial.attrs else None,
            'session': trial.attrs['session'], 
            'block_num': trial.attrs['block_num'], 
            'trial_num': trial.attrs['trial_num']
        }

        
        return item 
