import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
from bidict import bidict
from tqdm import tqdm

DATA_PATH = 'data/'
NUM_BARS = 8
RESOLUTION = 1 / 16
MAX_LENGTH = int(NUM_BARS/RESOLUTION)
NOTE_VALUE_SCALE = 1.
FILE_IDS = [str(i) for i in range(1024)]

"""
pitch_count = {}

for json_path in json_paths:
    with open(os.path.join(DATA_PATH, json_path), 'r') as f:
        raw = json.load(f)
        note_dict_list = raw['patterns'][0]['core']['notes']  # 0-lead, 1-chord, 2-bass, 3-sub

        for note_dict in note_dict_list:
            if note_dict['key_name'] not in pitch_count.keys():
                pitch_count[note_dict['key_name']] = 1
            else:
                pitch_count[note_dict['key_name']] += 1
print(len(pitch_count))
"""

class LeadNoteDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self._data = data
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        return self._data[index]
    

if __name__ == '__main__':
    PITCH2ID = bidict({'<r>': 0})  # the rest sign

    processed_data = []
    
    for file_id in tqdm(FILE_IDS, desc='processing'):
        json_path = f'track-{file_id}.json'
        mel_path_left = f'{file_id}_left.jpg'
        mel_path_right = f'{file_id}_right.jpg'
        
        with open(os.path.join(DATA_PATH, json_path), 'r') as f:
            raw = json.load(f)
            note_dict_list = raw['patterns'][0]['core']['notes']  # 0-lead, 1-chord, 2-bass, 3-sub
            
            pitches = [0] * MAX_LENGTH
            values = [0.] * MAX_LENGTH
            
            for note_dict in note_dict_list:
                key_name = note_dict['key_name']
                note_value = note_dict['note_value']
                pos_in_pattern = note_dict['pos_in_pattern']
                
                if key_name not in PITCH2ID.keys():
                    PITCH2ID[key_name] = len(PITCH2ID)
                i = int(pos_in_pattern/(RESOLUTION*4))  # assume 4/4
                pitches[i] = PITCH2ID[key_name]
                values[i] = note_value * NOTE_VALUE_SCALE
        
        mel_left_tensor = read_image(os.path.join(DATA_PATH, mel_path_left))
        mel_right_tensor = read_image(os.path.join(DATA_PATH, mel_path_right))
        
        processed_data += [(
            torch.LongTensor(pitches),
            torch.Tensor(values),
            mel_left_tensor,
            mel_right_tensor,
        )]
        
        torch.save(LeadNoteDataset(processed_data), os.path.join(DATA_PATH, 'processed.pt'))
            
            
        
                
# print(PITCH2ID)

