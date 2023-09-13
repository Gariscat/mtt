# Infer by SheetSage base version (without OpenAI Jukebox)
from pretty_midi import PrettyMIDI
from tqdm import tqdm
import json
from parse import *
from loopy.utils import piano_key2midi_id
import numpy as np
from typing import List
from sklearn.metrics import f1_score
import torch
from torch.utils.data import random_split
g = torch.Generator()
g.manual_seed(26)

bpm = 128
beat_per_sec = bpm / 60
scale = 4 # resolution is 1/16 while one beat is 1/4. 1/4 is 4 times 1/16.

def pos2hot(loc: List[int], correction: int = 0):
    ret = [0] * MAX_LENGTH
    for x in loc:
        pos = x + correction
        if 0 <= pos < MAX_LENGTH:
            ret[pos] = 1
    return ret

dataset = LeadNoteDataset(length=TOT_TRACK)
train_set, val_set = random_split(dataset, [0.8, 0.2], generator=g)
val_track_paths = [x[4] for x in val_set]
# print("ids in the validation set", val_track_paths)

for correction in (-2, -1, 0, 1, 2):
    all_onset_gt = []
    all_onset_pred = []
    
    for val_track_path in tqdm(val_track_paths):
        id = val_track_path[val_track_path.index('-')+1:val_track_path.index('.')]
        midi_data = PrettyMIDI(f'/root/SheetSage/output/{id}/output.midi')
        with open(f'data/track-{id}.json','r') as f:
            gt_json = json.load(f)
        
        assert len(midi_data.instruments) == 3
        assert midi_data.instruments[0].is_drum is True
        assert midi_data.instruments[1].notes[0].start == midi_data.instruments[1].notes[1].start # is_chord is True
        
        note_dict_list = gt_json['patterns'][0]['core']['notes']  # 0-lead, 1-chord, 2-bass, 3-sub
        pitche_gt = [0] * MAX_LENGTH
        attack_pred = [0] * MAX_LENGTH
        pitche_pred = [0] * MAX_LENGTH
        attack_pred = [0] * MAX_LENGTH
        
        attack_gt = pos2hot([int(note_dict['pos_in_pattern']*scale) for note_dict in note_dict_list if 'main' in note_dict['generator']])
        attack_pred = pos2hot([int(note.start*beat_per_sec*scale) for note in midi_data.instruments[2].notes], correction=correction) # 0-drum, 1-chord, 2-lead

        all_onset_gt += attack_gt
        all_onset_pred += attack_pred

        """print('------gt------')
        for note_dict in note_dict_list:
            if 'main' not in note_dict['generator']:
                continue
            print(piano_key2midi_id(note_dict['key_name']),
                note_dict['pos_in_pattern']*scale,
                note_dict['pos_in_pattern']*scale+note_dict['note_value']*16, # resolution is 1/16
                note_dict)
        print('------sh------')
        for note in midi_data.instruments[2].notes:
            print(int(note.start*beat_per_sec*scale), int(note.end*beat_per_sec*scale), note)
        break"""
        
    onset_f1 = f1_score(np.array(all_onset_gt), np.array(all_onset_pred))
    print("correction", correction*RESOLUTION, "onset_f1:", onset_f1)