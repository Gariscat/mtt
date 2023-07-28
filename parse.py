import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
from bidict import bidict
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt

DATA_PATH = '/root/mtt/data'
NUM_BARS = 8
RESOLUTION = 1 / 16  # prog-house
MAX_LENGTH = int(NUM_BARS/RESOLUTION)
NOTE_VALUE_SCALE = 1.
TOT_TRACK = 512
FILE_IDS = [str(i) for i in range(TOT_TRACK)]

"""
def transform(a: torch.Tensor):
    assert len(a.shape) == 3
    stride = a.shape[-1] // MAX_LENGTH
    C, H, W = a.shape
    ret = torch.cat((
        torch.cat((torch.zeros(C, H, stride), a[:, :, :-stride]), dim=-1),
        a,
        torch.cat((a[:, :, stride:], torch.zeros(C, H, stride)), dim=-1)
    ), dim=0)
    C_new, H_new, W_new = ret.shape
    assert C_new == 3 * C and H_new == H and W_new == W
    return ret
"""    
    
class LeadNoteDataset(Dataset):
    def __init__(self, length=TOT_TRACK) -> None:
        super().__init__()
        self._length = length
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        json_path = f'track-{index}.json'
        mel_path_left = f'{index}_left.jpg'
        mel_path_right = f'{index}_right.jpg'
        # read
        mel_left_tensor = read_image(os.path.join(DATA_PATH, mel_path_left)).float() / 255
        mel_right_tensor = read_image(os.path.join(DATA_PATH, mel_path_right)).float() / 255
        """
        # rgb-to-grayscale
        mel_left_tensor = rgb_to_grayscale(mel_left_tensor)
        mel_right_tensor = rgb_to_grayscale(mel_right_tensor)
        # include-3-frames-in-1-token
        mel_left_tensor = transform(mel_left_tensor)
        mel_right_tensor = transform(mel_right_tensor)
        """
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
            json_path
        )
    

# PITCH2ID = bidict({'<r>': 0})  # the rest sign
PITCH2ID = dict({'<r>': [0, TOT_TRACK * MAX_LENGTH]})
ATTACK_CNT = [TOT_TRACK * MAX_LENGTH, 0]

for file_id in tqdm(list(range(TOT_TRACK)), desc=f'scanning'):
    json_path = f'track-{file_id}.json'            
    with open(os.path.join(DATA_PATH, json_path), 'r') as f:
        raw = json.load(f)
        note_dict_list = raw['patterns'][0]['core']['notes']  # 0-lead, 1-chord, 2-bass, 3-sub
        
        for note_dict in note_dict_list:
            if 'main' not in note_dict['generator']:
                continue
            key_name = note_dict['key_name']
            note_value = note_dict['note_value']
            pos_in_pattern = note_dict['pos_in_pattern']
            
            cur_cnt = note_value // RESOLUTION
                    
            if key_name not in PITCH2ID.keys():
                PITCH2ID[key_name] = [len(PITCH2ID), cur_cnt]
            else:
                PITCH2ID[key_name][1] += cur_cnt
            
            PITCH2ID['<r>'][1] -= cur_cnt
            ATTACK_CNT[0] -= 1
            ATTACK_CNT[1] += 1
        # torch.save(LeadNoteDataset(processed_data), os.path.join(DATA_PATH, f'processed-{st}-{ed}.pt'))

ID2PITCH = dict()
for pitch, (id, cnt) in PITCH2ID.items():
    ID2PITCH[id] = pitch
print(PITCH2ID)