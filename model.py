from vit_pytorch import SimpleViT, ViT
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.models import resnet18
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LSTM, GRU, RNN
import pytorch_lightning as pl
from parse import MAX_LENGTH, PITCH2ID, ID2PITCH, ATTACK_CNT
import os, shutil, io
import numpy as np
import loopy
from loopy.utils import midi_id2piano_key, piano_key2midi_id
from typing import List
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import wandb
from sklearn import metrics

class LeadModel(pl.LightningModule):
    def __init__(self,
        config: dict,
        opt_name: str = 'Adam',
        lr: float = 1e-3,
        loss_alpha: float = 0.8,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        
        if 'tmp' in os.listdir():
            shutil.rmtree('tmp')
        
        self.opt_name = opt_name
        self.lr = lr
        self.image_size = config['image_size']
        self.loss_alpha = loss_alpha
        self.is_causal = is_causal
        self.rnn_type = config['rnn_type']
        self.pitch_loss_w = 1 / torch.Tensor([cnt for id, cnt in PITCH2ID.values()])
        # print("\n\npitch loss weight:", self.pitch_loss_w, '\n\n')
        self.attack_loss_w = 1 / torch.Tensor(ATTACK_CNT)
        # print("\n\nattack loss weight:", self.attack_loss_w, '\n\n')
        ext_name = str(config['extractor_name'])
        self.extractor_backbone = None
        
        if 'ViT' in ext_name:
            vit_cls = None
            if ext_name == 'SimpleViT':
                vit_cls = SimpleViT
            elif ext_name == 'ViT':
                vit_cls = ViT
            self.extractor_backbone = vit_cls(
                image_size=config['image_size'],  # 128
                patch_size=config['patch_size'],  # 4
                num_classes=config['num_classes'],  # 1000
                dim=config['dim'],  # 1024
                depth=config['depth'],  # 6
                heads=config['heads'],  # 16
                mlp_dim=config['mlp_dim'],  # 2048
                channels=6,  # 2*3
            )   
        elif ext_name == 'ResNet18':
            self.extractor_backbone = resnet18(weights='IMAGENET1K_V1')
        """
        elif ext_name == 'ResNet18':
            from torchvision.models import resnet18
            self.extractor_backbone = resnet18(weights=config['weights'])  # IMAGENET1K_V1
        """
        if 'vit' in str(type(self.extractor_backbone)):
            self.extractor = nn.Sequential(
                self.extractor_backbone,
                nn.Dropout(config['dropout']),
                nn.Linear(config['num_classes'], config['out_dim'])
            )
        elif self.extractor_backbone is not None:
            self.extractor = nn.Sequential(
                nn.Conv2d(6, 3, kernel_size=1),
                self.extractor_backbone,
                nn.Dropout(config['dropout']),
                nn.Linear(config['num_classes'], config['out_dim'])
            )
        elif config['hidden_size'] is not None:
            flattened_dim = 6 * config['image_size'] * config['image_size'] // MAX_LENGTH
            hidden_size = config['hidden_size']
            self.extractor = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(flattened_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, config['out_dim'])
            )
        else:  # None
            flattened_dim = 6 * config['image_size'] * config['image_size'] // MAX_LENGTH
            self.extractor = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(flattened_dim, config['out_dim'])
            )
        
        if config['rnn_type'] is None:
            encoder_layer = TransformerEncoderLayer(
                d_model=config['out_dim'],
                nhead=config['nhead'],
                batch_first=True,
            )
            self.main_model = TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
            
        else:
            model_cls = RNN
            if config['rnn_type'] == 'lstm':
                model_cls = LSTM
            elif config['rnn_type'] == 'gru':
                model_cls = GRU
                
            self.main_model = model_cls(
                input_size=config['out_dim'],
                hidden_size=config['out_dim'],
                num_layers=config['num_layers'],
                batch_first=True,
                bidirectional=config['bidirectional']
            )
        
        self.pitch_linear_layer = nn.Linear(config['out_dim'], len(PITCH2ID))
        self.attack_linear_layer = nn.Linear(config['out_dim'], 2)
        
    def forward(self, input_tensor):  # (B, C, H, W)
        seq_len = MAX_LENGTH
        ### input_tensor = F.interpolate(input=input_tensor, size=(self.image_size, 32 * seq_len))
        B, C, H, W = input_tensor.size()
        W //= seq_len
        input_tensor = input_tensor.reshape(B, C, H, W, seq_len).permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)
        # (B, C, H, W*M) -> (B, C, H, W, M) -> (B, M, C, H, W) -> (B*M, C, H, W)
        extracted = self.extractor(input_tensor)  # (B*M, O)
        extracted = extracted.reshape(B, seq_len, -1)  # (B, M, O)
        
        if self.rnn_type is None:
            hidden_states = self.main_model(extracted, is_causal=self.is_causal)
        else:
            hidden_states, (_, __) = self.main_model(extracted)
            
        pitch_logits = self.pitch_linear_layer(hidden_states)
        attack_logits = self.attack_linear_layer(hidden_states)
        return pitch_logits, attack_logits
    
    def configure_optimizers(self):
        opt_cls = None
        if self.opt_name == 'Adam':
            opt_cls = optim.Adam
        elif self.opt_name == 'AdamW':
            opt_cls = optim.AdamW
        else:
            opt_cls = optim.SGD
        return opt_cls(self.parameters(), lr=self.lr)
    
    def training_step(self, train_batch, *args, **kwargs):
        pitch_gt, attack_gt, mel_left_tensor, mel_right_tensor, json_path = train_batch
        mel_tensor = torch.cat((mel_left_tensor, mel_right_tensor), dim=1)

        pitch_logits, attack_logits = self.forward(mel_tensor)
        self.pitch_loss_w = self.pitch_loss_w.to(pitch_gt.device)
        self.attack_loss_w = self.attack_loss_w.to(attack_gt.device)
        pitch_loss_func = nn.CrossEntropyLoss(weight=self.pitch_loss_w)
        attack_loss_func = nn.CrossEntropyLoss(weight=self.attack_loss_w)
        
        pitch_pred = pitch_logits.argmax(-1)
        attack_pred = attack_logits.argmax(-1)
        """
        tmp = pitch_gt[0].cpu().numpy().tolist()
        print("GT:", [ID2PITCH[_] for _ in tmp])
        tmp = pitch_pred[0].cpu().numpy().tolist()
        print("pred:", [ID2PITCH[_] for _ in tmp])
        """
        # self.visualize_pitch(pitch_gt[0], pitch_pred[0], 'train')
        
        pitch_logits = pitch_logits.reshape(-1, len(PITCH2ID))
        pitch_gt = pitch_gt.flatten()
        attack_logits = attack_logits.reshape(-1, 2)
        attack_gt = attack_gt.flatten()
        
        pitch_loss = pitch_loss_func(pitch_logits, pitch_gt)
        attack_loss = attack_loss_func(attack_logits, attack_gt)
        loss = self.loss_alpha * pitch_loss + (1 - self.loss_alpha) * attack_loss
        self.log('train_loss_pitch', pitch_loss)
        self.log('train_loss_attack', attack_loss)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, *args, **kwargs):
        pitch_gt, attack_gt, mel_left_tensor, mel_right_tensor, json_path = val_batch
        mel_tensor = torch.cat((mel_left_tensor, mel_right_tensor), dim=1)
        
        os.makedirs('tmp', exist_ok=True)
        cnt = [0] * len(PITCH2ID)
        stride = mel_left_tensor.shape[-1] // MAX_LENGTH
        
        """plt.imshow(mel_left_tensor[0].permute(1, 2, 0).numpy())
        plt.savefig('tmp.jpg')
        plt.close()
        print(json_path)"""
        
        for j in range(MAX_LENGTH):
            pitch_id = pitch_gt[0][j].item()
            mel_left_cur = mel_left_tensor[0, :, :, j*stride:(j+1)*stride].cpu()
        
            # mel_left_cur = F.interpolate(mel_left_cur, (512, 4))
            # mel_left_cur = torch.cat([mel_left_cur[i] for i in range(mel_left_cur.shape[0])], dim=-1)
            # save_image(mel_left_cur, f'tmp/{ID2PITCH[pitch_id]}-{cnt[pitch_id]}.jpg')
            ### save_image(mel_left_cur, f'tmp/{j}-{ID2PITCH[pitch_id]}.jpg')
            cnt[pitch_id] += 1
        """"""     
        pitch_logits, attack_logits = self.forward(mel_tensor)
        self.pitch_loss_w = self.pitch_loss_w.to(pitch_gt.device)
        self.attack_loss_w = self.attack_loss_w.to(attack_gt.device)
        pitch_loss_func = nn.CrossEntropyLoss(weight=self.pitch_loss_w)
        attack_loss_func = nn.CrossEntropyLoss(weight=self.attack_loss_w)
        
        pitch_pred = pitch_logits.argmax(-1)
        attack_pred = attack_logits.argmax(-1)
        
        # tmp = pitch_gt[0].cpu().numpy().tolist()
        # print("GT:", [ID2PITCH[_] for _ in tmp])
        # tmp = pitch_pred[0].cpu().numpy().tolist()
        # print("pred:", [ID2PITCH[_] for _ in tmp])
        
        for i in range(pitch_logits.shape[0]):
            if np.random.rand() > 0.5: # visualize only half the results to speed up training
                continue
            if self.loss_alpha == 1: # pitch only
                self.visualize_pitch(pitch_gt[i], pitch_pred[i], 'val')
            elif self.loss_alpha == 0: # onset only
                self.visualize_onset(attack_gt[i], attack_pred[i], 'val')
            """    
            print('Pitch-GT:', pitch_gt[i])
            print('Pitch-PD:', pitch_pred[i])
            print('Attack-GT:', attack_gt[i])
            print('Attack-PD:', attack_pred[i])
            """
        
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(
            y_true=attack_gt.flatten().cpu().numpy(),
            y_pred=attack_pred.flatten().cpu().numpy(),
        )
        """print(precision)
        print(recall)
        print(fscore)
        exit()"""
        self.log('val_attack_precision', precision[1])
        self.log('val_attack_recall', recall[1])
        self.log('val_attack_fscore', fscore[1])
        
        pitch_logits = pitch_logits.reshape(-1, len(PITCH2ID))
        pitch_gt = pitch_gt.flatten()
        attack_logits = attack_logits.reshape(-1, 2)
        attack_gt = attack_gt.flatten()
        
        pitch_loss = pitch_loss_func(pitch_logits, pitch_gt)
        attack_loss = attack_loss_func(attack_logits, attack_gt)
        loss = self.loss_alpha * pitch_loss + (1 - self.loss_alpha) * attack_loss
        self.log('val_loss_pitch', pitch_loss)
        self.log('val_loss_attack', attack_loss)
        self.log('val_loss', loss)

    def visualize_pitch(self, gt: torch.Tensor, pred: torch.Tensor, phase: str):
        def plot(lst: List[int], ax: matplotlib.axes.Axes, title: str=None):
            segments = []
            for i, v in enumerate(lst):
                if ID2PITCH[v] == '<r>':
                    continue
                pitch = piano_key2midi_id(ID2PITCH[v])
                segments += [((i, pitch), (i+1, pitch))]
            ax.add_collection(LineCollection(segments))
            ax.autoscale()
            m = ax.get_yticks().tolist()
            for i in range(len(m)):
                m[i] = midi_id2piano_key(int(m[i]))
            ax.set_yticklabels(m)
            ax.set_title(title)
                
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        gt = gt.detach().cpu().numpy().tolist()
        pred = pred.detach().cpu().numpy().tolist()
        plot(gt, axes[0], 'ground truth')
        plot(pred, axes[1], 'prediction')

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        if phase == 'val':
            self.logger.log_image(key=f"{phase}_pitch_samples", images=[img,])
        elif phase == 'train':
            if np.random.rand() >= 0.9:
                self.logger.log_image(key=f"{phase}_pitch_samples", images=[img,])
        plt.close()
        
    def visualize_onset(self, gt: torch.Tensor, pred: torch.Tensor, phase: str):
        def plot(lst: List[int], ax: matplotlib.axes.Axes, title: str=None):
            ax.set_title(title)
            for i, x in enumerate(lst):
                if x == 0:
                    continue
                ax.vlines(i, 0, 1, color='r' if 'pred' in title else 'g', alpha=0.9,
                    linestyle='--', label='onsets'
                )
            ax.legend()
                
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        gt = gt.detach().cpu().numpy().tolist()
        pred = pred.detach().cpu().numpy().tolist()
        plot(gt, axes[0], 'ground truth')
        plot(pred, axes[1], 'prediction')

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        if phase == 'val':
            self.logger.log_image(key=f"{phase}_onset_samples", images=[img,])
        elif phase == 'train':
            if np.random.rand() >= 0.9:
                self.logger.log_image(key=f"{phase}_onset_samples", images=[img,])
        plt.close()
