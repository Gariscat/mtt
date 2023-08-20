from model import *
from torch.utils.data import DataLoader, random_split
from parse import *
import argparse
import wandb
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
g = torch.Generator()
g.manual_seed(26)


def plot(
    pitch_lst: List[int],
    onset_lst: List[int],
    ax: matplotlib.axes.Axes,
    title: str = None
):
    segments = []
    for i, v in enumerate(pitch_lst):
        if ID2PITCH[v] == '<r>':
            continue
        pitch = piano_key2midi_id(ID2PITCH[v])
        segments += [((i, pitch), (i+1, pitch))]
    ax.add_collection(LineCollection(segments))
    
    for i, x in enumerate(onset_lst):
        if x == 0:
            continue
        pitch = piano_key2midi_id(ID2PITCH[pitch_lst[i]])
        ax.vlines(i, pitch-0.5, pitch+0.5, color='r' if 'pred' in title else 'g', alpha=0.9, linestyle='solid')
    
    ax.autoscale()
    m = ax.get_yticks().tolist()
    for i in range(len(m)):
        m[i] = midi_id2piano_key(int(m[i]))
    ax.set_yticklabels(m)
    ax.set_title(title)


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
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--opt_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--loss_alpha', type=float, default=0.5)
    # transformer
    parser.add_argument('--is_causal', type=bool, default=False)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
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
    
    wandb.init(
        entity='gariscat',
        project='MTTLeadJointInference',
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pitch_config = deepcopy(config)
    pitch_config['num_layers'] = 3
    pitch_model = LeadModel.load_from_checkpoint(
        '/root/mtt/ckpt/MTTLeadAdamWPitchVis/uzj0xbhy/checkpoints/epoch=49-step=100000.ckpt',
        config=pitch_config,
        loss_alpha=1
    ).to(device)
    onset_config = deepcopy(config)
    onset_config['num_layers'] = 1
    onset_model = LeadModel.load_from_checkpoint(
        '/root/mtt/ckpt/MTTLeadAdamWOnsetVis/by7630b0/checkpoints/epoch=21-step=44000.ckpt',
        config=onset_config,
        loss_alpha=0,
    ).to(device)
    print(pitch_model.is_causal, onset_model.is_causal)
    dataset = LeadNoteDataset(length=args.dataset_length)
    
    train_set, val_set = random_split(dataset, [0.8, 0.2], generator=g)
    # train_loader = DataLoader(dataset=train_set, batch_size=1,)
    val_loader = DataLoader(dataset=val_set, batch_size=1,)
    
    print(ID2PITCH)
    
    for batch in tqdm(val_loader):
        pitch_gt, onset_gt, mel_left, mel_right, json_path = batch
        mel_tensor = torch.cat((mel_left, mel_right), dim=1).to(device)
        
        pitch_logits, _ = pitch_model.forward(mel_tensor)
        _, onset_logits = onset_model.forward(mel_tensor)
        
        pitch_pred = pitch_logits.argmax(-1)
        onset_pred = onset_logits.argmax(-1)
        
        pitch_gt = pitch_gt.flatten().numpy().tolist()
        onset_gt = onset_gt.flatten().numpy().tolist()
        pitch_pred = pitch_pred.detach().cpu().flatten().numpy().tolist()
        onset_pred = onset_pred.detach().cpu().flatten().numpy().tolist()
        
        # validate the notes
        
        L = len(pitch_gt)
        
        for i in range(L):
            if onset_pred[i]:
                if pitch_pred[i] == 0:
                    onset_pred[i] = 0
            else:
                if pitch_pred[i]:
                    if i == 0 or pitch_pred[i-1] != pitch_pred[i]:
                        pitch_pred[i] = 0
                        
        # plot
        
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        plot(pitch_gt, onset_gt, axes[0], 'ground truth')
        plot(pitch_pred, onset_pred, axes[1], 'prediction')
        plt.suptitle(json_path)
        
        # log
        
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        # print(type(img))
        img = wandb.Image(img)
        # print(type(img))
        wandb.log({"joint_inference_samples": img})
        plt.close()