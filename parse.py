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
FILE_IDS = [str(i) for i in range(512)]

"""
pitch_count = {}

for id in FILE_IDS:
    with open(os.path.join(DATA_PATH, f'track-{id}.json'), 'r') as f:
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
    def __init__(self, length=512) -> None:
        super().__init__()
        self._length = length
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        json_path = f'track-{index}.json'
        mel_path_left = f'{index}_left.jpg'
        mel_path_right = f'{index}_right.jpg'
        
        mel_left_tensor = read_image(os.path.join(DATA_PATH, mel_path_left)).float() / 255
        mel_right_tensor = read_image(os.path.join(DATA_PATH, mel_path_right)).float() / 255
        
        with open(os.path.join(DATA_PATH, json_path), 'r') as f:
            raw = json.load(f)
            note_dict_list = raw['patterns'][0]['core']['notes']  # 0-lead, 1-chord, 2-bass, 3-sub
            
            pitches = [0] * MAX_LENGTH
            values = [0.] * MAX_LENGTH
                
            for note_dict in note_dict_list:
                key_name = note_dict['key_name']
                note_value = note_dict['note_value']
                pos_in_pattern = note_dict['pos_in_pattern']
                    
                i = int(pos_in_pattern/(RESOLUTION*4))  # assume 4/4
                pitches[i] = PITCH2ID[key_name]
                values[i] = note_value * NOTE_VALUE_SCALE
        
        return (
            torch.LongTensor(pitches),
            torch.Tensor(values),
            mel_left_tensor,
            mel_right_tensor,
        )
    

PITCH2ID = bidict({'<r>': 0})  # the rest sign

for i in range(1024//64):
    st, ed = i*64, (i+1)*64
    FILE_IDS = list(range(st, ed))
    processed_data = []
    for file_id in tqdm(FILE_IDS, desc=f'processing {st}-{ed}'):
        json_path = f'track-{file_id}.json'            
        with open(os.path.join(DATA_PATH, json_path), 'r') as f:
            raw = json.load(f)
            note_dict_list = raw['patterns'][0]['core']['notes']  # 0-lead, 1-chord, 2-bass, 3-sub
                
            for note_dict in note_dict_list:
                key_name = note_dict['key_name']
                note_value = note_dict['note_value']
                pos_in_pattern = note_dict['pos_in_pattern']
                    
                if key_name not in PITCH2ID.keys():
                    PITCH2ID[key_name] = len(PITCH2ID)
        # torch.save(LeadNoteDataset(processed_data), os.path.join(DATA_PATH, f'processed-{st}-{ed}.pt'))
                
                