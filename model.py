from vit_pytorch import SimpleViT
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import pytorch_lightning as pl
from parse import MAX_LENGTH, PITCH2ID, ID2PITCH
import os, shutil
import numpy as np

class LeadModel(pl.LightningModule):
    def __init__(self,
        backbone_config: dict,
        transformer_config: dict(),
        opt_name: str = 'Adam',
        lr: float = 1e-3,
        loss_alpha: float = 0.8,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        
        shutil.rmtree('tmp')
        
        self.opt_name = opt_name
        self.lr = lr
        self.image_size = backbone_config['image_size']
        self.loss_alpha = loss_alpha
        self.is_causal = is_causal
        """self.loss_weight = torch.Tensor([cnt for id, cnt in PITCH2ID.values()])
        self.loss_weight = -1/torch.log(self.loss_weight)
        print("\n\n", self.loss_weight, '\n\n')
        self.loss_weight = torch.softmax(self.loss_weight, dim=0)
        print("\n\n", self.loss_weight, '\n\n')"""
        
        if backbone_config['extractor_name'] == 'SimpleViT':
            self.extractor_backbone = SimpleViT(
                image_size=backbone_config['image_size'],  # 128
                patch_size=backbone_config['patch_size'],  # 4
                num_classes=backbone_config['num_classes'],  # 1000
                dim=backbone_config['dim'],  # 1024
                depth=backbone_config['depth'],  # 6
                heads=backbone_config['heads'],  # 16
                mlp_dim=backbone_config['mlp_dim'],  # 2048
                channels=6,  # 2*3
            )   
        else:
            self.extractor_backbone = None
        """
        elif backbone_config['extractor_name'] == 'ResNet18':
            from torchvision.models import resnet18
            self.extractor_backbone = resnet18(weights=backbone_config['weights'])  # IMAGENET1K_V1
        """
        if 'vit' in str(type(self.extractor_backbone)):
            self.extractor = nn.Sequential(
                self.extractor_backbone,
                nn.Dropout(backbone_config['dropout']),
                nn.Linear(backbone_config['num_classes'], backbone_config['out_dim'])
            )
        elif self.extractor_backbone is not None:
            self.extractor = nn.Sequential(
                nn.Conv2d(6, 3, kernel_size=1),
                self.extractor_backbone,
                nn.Dropout(backbone_config['dropout']),
                nn.Linear(backbone_config['num_classes'], backbone_config['out_dim'])
            )
        else:
            flattened_dim = 6 * backbone_config['image_size'] * backbone_config['image_size'] // MAX_LENGTH
            self.extractor = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(flattened_dim, backbone_config['out_dim'])
            )
        
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_config['d_model'],
            nhead=transformer_config['nhead'],
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=transformer_config['num_layers'])
        self.pitch_linear_layer = nn.Linear(transformer_config['d_model'], len(PITCH2ID))
        self.attack_linear_layer = nn.Linear(transformer_config['d_model'], 2)
        
    def forward(self, input_tensor):  # (B, C, H, W)
        seq_len = MAX_LENGTH
        ### input_tensor = F.interpolate(input=input_tensor, size=(self.image_size, 32 * seq_len))
        B, C, H, W = input_tensor.size()
        W //= seq_len
        input_tensor = input_tensor.reshape(B, C, H, W, seq_len).permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)
        # (B, C, H, W*M) -> (B, C, H, W, M) -> (B, M, C, H, W) -> (B*M, C, H, W)
        extracted = self.extractor(input_tensor)  # (B*M, O)
        extracted = extracted.reshape(B, seq_len, -1)  # (B, M, O)
        
        hidden_states = self.transformer(extracted, is_causal=self.is_causal)
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
        pitch_gt, attack_gt, mel_left_tensor, mel_right_tensor = train_batch
        mel_tensor = torch.cat((mel_left_tensor, mel_right_tensor), dim=1)

        pitch_logits, attack_logits = self.forward(mel_tensor)
        # self.loss_weight = self.loss_weight.to(pitch_gt.device)
        pitch_loss_func = nn.CrossEntropyLoss(weight=None)
        attack_loss_func = nn.CrossEntropyLoss(weight=None)
        
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
        pitch_gt, attack_gt, mel_left_tensor, mel_right_tensor = val_batch
        mel_tensor = torch.cat((mel_left_tensor, mel_right_tensor), dim=1)
        """
        os.makedirs('tmp', exist_ok=True)
        cnt = [0] * len(PITCH2ID)
        stride = mel_left_tensor.shape[-1] // MAX_LENGTH
        for j in range(MAX_LENGTH):
            pitch_id = pitch_gt[0][j].item()
            mel_left_cur = mel_left_tensor[:, :, :, j*stride:(j+1)*stride].cpu()
            # print(mel_left_cur.shape)
            mel_left_cur = F.interpolate(mel_left_cur, (256, 64))[0]
            mel_left_cur = torch.cat([mel_left_cur[i] for i in range(mel_left_cur.shape[0])], dim=-1)
            save_image(mel_left_cur, f'tmp/{ID2PITCH[pitch_id]}-{cnt[pitch_id]}.jpg')
            cnt[pitch_id] += 1
        """
        pitch_logits, attack_logits = self.forward(mel_tensor)
        # self.loss_weight = self.loss_weight.to(pitch_gt.device)
        pitch_loss_func = nn.CrossEntropyLoss(weight=None)
        attack_loss_func = nn.CrossEntropyLoss(weight=None)
        
        pitch_pred = pitch_logits.argmax(-1)
        attack_pred = attack_logits.argmax(-1)
        
        for i in range(pitch_logits.shape[0]):
            print('Pitch-GT:', pitch_gt[i])
            print('Pitch-PD:', pitch_pred[i])
            print('Attack-GT:', attack_gt[i])
            print('Attack-PD:', attack_pred[i])
        """"""
        
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
