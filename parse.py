import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
from bidict import bidict
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH = 'data/'
NUM_BARS = 8
RESOLUTION = 1 / 16  # prog-house
MAX_LENGTH = int(NUM_BARS/RESOLUTION)
NOTE_VALUE_SCALE = 1.
TOT_TRACK = 512
FILE_IDS = [str(i) for i in range(TOT_TRACK)]


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
            attacks = [0] * MAX_LENGTH
                
            for note_dict in note_dict_list:
                if 'main' not in note_dict['generator']:
                    continue
                key_name = note_dict['key_name']
                note_value = note_dict['note_value']
                pos_in_pattern = note_dict['pos_in_pattern']
                # assume 4/4 signature
                i = int(pos_in_pattern/(RESOLUTION*4))
                j = i + int(note_value/RESOLUTION)
                assert i < j <= MAX_LENGTH
                for _ in range(i, j):
                    pitches[_] = PITCH2ID[key_name][0]
                """pitches[i] = PITCH2ID[key_name][0]"""
                attacks[i] = 1
        
        return (
            torch.LongTensor(pitches),
            torch.LongTensor(attacks),
            mel_left_tensor,
            mel_right_tensor,
        )
    

# PITCH2ID = bidict({'<r>': 0})  # the rest sign
PITCH2ID = dict({'<r>': [0, TOT_TRACK * MAX_LENGTH]})

for file_id in tqdm(list(range(TOT_TRACK)), desc=f'scanning'):
    json_path = f'track-{file_id}.json'            
    with open(os.path.join(DATA_PATH, json_path), 'r') as f:
        raw = json.load(f)
        note_dict_list = raw['patterns'][0]['core']['notes']  # 0-lead, 1-chord, 2-bass, 3-sub
        
        note_cnt = 0
        for note_dict in note_dict_list:
            if 'main' not in note_dict['generator']:
                continue
            key_name = note_dict['key_name']
            note_value = note_dict['note_value']
            pos_in_pattern = note_dict['pos_in_pattern']
                    
            if key_name not in PITCH2ID.keys():
                PITCH2ID[key_name] = [len(PITCH2ID), 1]
            else:
                PITCH2ID[key_name][1] += 1
            
            note_cnt += 1
            
        PITCH2ID['<r>'][1] -= note_cnt
        # torch.save(LeadNoteDataset(processed_data), os.path.join(DATA_PATH, f'processed-{st}-{ed}.pt'))

print(PITCH2ID)