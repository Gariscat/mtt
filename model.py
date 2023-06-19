from vit_pytorch import SimpleViT
import torch
from torch import nn, optim
import torch.nn.functional as F
from transformer import *
import pytorch_lightning as pl
from parse import MAX_LENGTH


class LeadModel(pl.LightningModule):
    def __init__(self,
        backbone_config: dict,
        transformer_config: TransformerConfig = TransformerConfig(),
        opt_name: str = 'Adam',
        lr: float = 1e-3,
        loss_alpha: float = 0.8,
    ) -> None:
        super().__init__()
        
        assert backbone_config['out_dim'] == transformer_config.input_size
        
        self.opt_name = opt_name
        self.lr = lr
        self.image_size = backbone_config['image_size']
        self.loss_alpha = loss_alpha
        
        if backbone_config['extractor_name'] == 'SimpleViT':
            self.extractor_backbone = SimpleViT(
                image_size=backbone_config['image_size'],  # 128
                patch_size=backbone_config['patch_size'],  # 4
                num_classes=backbone_config['num_classes'],  # 512
                dim=backbone_config['dim'],  # 1024
                depth=backbone_config['depth'],  # 6
                heads=backbone_config['heads'],  # 16
                mlp_dim=backbone_config['mlp_dim'],  # 2048
                channels=6,  # 2*3
            )   
        else:
            raise NotImplementedError(f"{backbone_config['extractor_name']} is not supported as backbone")
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
        else:
            self.extractor = nn.Sequential(
                nn.Conv2d(6, 3, kernel_size=1),
                self.extractor_backbone,
                nn.Dropout(backbone_config['dropout']),
                nn.Linear(backbone_config['num_classes'], backbone_config['out_dim'])
            )
        
        self.transformer = TransformerLM(transformer_config)
        
    def forward(self, input_tensor):  # (B, C, H, W)
        input_tensor = F.interpolate(input=input_tensor, size=(self.image_size, 32 * MAX_LENGTH))
        B, C, H, W = input_tensor.size()
        N = W // H
        input_tensor = input_tensor.reshape(B, C, H, H, -1).permute(0, 4, 1, 2, 3).reshape(-1, C, H, H)
        # (B, C, H, W) -> (B, C, H, H, N) -> (B, N, C, H, H) -> (B*N, C, H, H)
        extracted = self.extractor(input_tensor)  # (B*N, O)
        extracted = extracted.reshape(B, N, -1)  # (B, N, O)
        """print(extracted.size())
        exit()"""
        
        pitch_logits, value_estimates, _ = self.transformer(extracted)
        return pitch_logits, value_estimates
    
    def configure_optimizers(self):
        opt_cls = optim.Adam if self.opt_name == 'Adam' else optim.SGD
        return opt_cls(self.parameters(), lr=self.lr)
    
    def training_step(self, train_batch, *args, **kwargs):
        pitch_gt, value_gt, mel_left_tensor, mel_right_tensor = train_batch
        mel_tensor = torch.cat((mel_left_tensor, mel_right_tensor), dim=1)

        pitch_logits, value_estimates = self.forward(mel_tensor)
        pitch_loss_func = nn.CrossEntropyLoss()
        value_loss_func = nn.MSELoss()
        
        pitch_logits = pitch_logits.reshape(-1, self.transformer.config.vocab_size)
        pitch_gt = pitch_gt.flatten()
        value_estimates = value_estimates.flatten()
        value_gt = value_gt.flatten()
        
        loss = self.loss_alpha * pitch_loss_func(pitch_logits, pitch_gt) + (1 - self.loss_alpha) * value_loss_func(value_estimates, value_gt)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, *args, **kwargs):
        pitch_gt, value_gt, mel_left_tensor, mel_right_tensor = val_batch
        mel_tensor = torch.cat((mel_left_tensor, mel_right_tensor), dim=1)

        pitch_logits, value_estimates = self.forward(mel_tensor)
        pitch_loss_func = nn.CrossEntropyLoss()
        value_loss_func = nn.MSELoss()
        
        # print(pitch_logits.size(), pitch_gt.size(), value_estimates.size(), value_gt.size())
        
        pitch_logits = pitch_logits.reshape(-1, self.transformer.config.vocab_size)
        pitch_gt = pitch_gt.flatten()
        value_estimates = value_estimates.flatten()
        value_gt = value_gt.flatten()
        
        loss = self.loss_alpha * pitch_loss_func(pitch_logits, pitch_gt) + (1 - self.loss_alpha) * value_loss_func(value_estimates, value_gt)
        self.log('train_loss', loss)
