from model import *
import wandb
import argparse
from pytorch_lightning.loggers import WandbLogger
import os
from parse import DATA_PATH, LeadNoteDataset
from torch.utils.data import DataLoader, random_split
g = torch.Generator()
g.manual_seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=512)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--mlp_dim', type=int, default=512)
    
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--out_dim', type=int, default=96)
    parser.add_argument('--extractor_name', type=str, default='SimpleViT')
    
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--opt_name', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss_alpha', type=float, default=0.8)
    
    args = parser.parse_args()
    backbone_config = vars(args)
    transformer_config = TransformerConfig(input_size=args.out_dim)
    
    model = LeadModel(
        backbone_config=backbone_config,
        transformer_config=transformer_config,
        opt_name=args.opt_name,
        lr=args.lr,
        loss_alpha=args.loss_alpha
    )
    
    dataset = torch.load(os.path.join(DATA_PATH, 'processed.pt'))
    train_set, val_set = random_split(dataset, [0.8, 0.2], generator=g)
    train_loader = DataLoader(dataset=train_set, batch_size=2,)
    val_loader = DataLoader(dataset=val_set, batch_size=2,)

    wandb_logger = WandbLogger(
        entity='gariscat',
        project='MTTLead',
        config=backbone_config,
        log_model=True,
        save_dir='./ckpt',
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices="auto",
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        deterministic=True,
        default_root_dir='./ckpt',
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    