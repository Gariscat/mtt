from model import *
import wandb
import argparse
from pytorch_lightning.loggers import WandbLogger
import os
from parse import *
from torch.utils.data import DataLoader, random_split, ConcatDataset
g = torch.Generator()
g.manual_seed(26)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ViT-extractor
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--mlp_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--out_dim', type=int, default=512)
    parser.add_argument('--extractor_name', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    # training
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--opt_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--loss_alpha', type=float, default=0.5)
    # transformer
    parser.add_argument('--is_causal', type=bool, default=False)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    # rnn
    parser.add_argument('--rnn_type', type=str, default=None)
    parser.add_argument('--bidirectional', type=bool, default=False)
    # misc
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--project_name', type=str, default='MTTLead')
    parser.add_argument('--dataset_length', type=int, default=5000)
    args = parser.parse_args()
    args.dataset_length = TOT_TRACK
    print(args)
    config = vars(args)
    
    model = LeadModel(
        config=config,
        opt_name=args.opt_name,
        lr=args.lr,
        loss_alpha=args.loss_alpha,
        is_causal=args.is_causal
    )
    
    dataset = LeadNoteDataset(length=args.dataset_length)
    
    train_set, val_set = random_split(dataset, [0.8, 0.2], generator=g)
    train_loader = DataLoader(dataset=train_set, batch_size=2,)
    val_loader = DataLoader(dataset=val_set, batch_size=2,)

    logger = WandbLogger(
        entity='gariscat',
        project=args.project_name,
        config=config,
        log_model=True,
        save_dir='./ckpt',
    ) if not args.debug else None
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else 'cpu',
        logger=logger,
        max_epochs=args.max_epochs,
        deterministic=True,
        default_root_dir='./ckpt',
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # model.validation_step(next(iter(val_loader)))
