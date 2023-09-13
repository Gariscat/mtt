import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from parse import *
from model import LeadModel
from joint_infer import pitch_ckpt, onset_ckpt, plot
import wandb
import io

def get_mel(y: np.ndarray, sr: int):
    for i, part in enumerate(('left', 'right')):
        """mel_spec = librosa.feature.melspectrogram(y=y[i], sr=sr)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)"""
        S = np.abs(librosa.stft(y[i], n_fft=4096))**2
        fig, ax = plt.subplots(nrows=1, sharex=True)
        img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr,
            y_axis='log', x_axis='time', ax=ax)
        plt.axis('off')
        plt.savefig('tmp.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
        ### plt.show()
        plt.close()
            
        tmp_img = Image.open('tmp.jpg')
        img = tmp_img.resize((512, 512))
        # img = ImageOps.flip(img)
        img.save('/root/tmp'+f'_{part}.jpg')
        

if __name__ == '__main__':
    y, sr = librosa.load('/root/forever.wav', mono=False)
    print(y.shape)
    get_mel(y, sr)
    
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
    parser.add_argument('--num_layers', type=int, default=4)
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
        project='MTTLeadInferNewMMM',
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pitch_model = LeadModel.load_from_checkpoint(
        pitch_ckpt,
        config=config,
        loss_alpha=1
    ).to(device)
    onset_model = LeadModel.load_from_checkpoint(
        onset_ckpt,
        config=config,
        loss_alpha=0,
    ).to(device)
    
    mel_path_left = '/root/tmp_left.jpg'
    mel_path_right = '/root/tmp_right.jpg'
    mel_left_tensor = read_image(mel_path_left).float() / 255
    mel_right_tensor = read_image(mel_path_right).float() / 255
    
    mel_tensor = torch.cat((
        mel_left_tensor.unsqueeze(0),
        mel_right_tensor.unsqueeze(0))
    , dim=1).to(device)
    
    pitch_logits, _ = pitch_model.forward(mel_tensor)
    _, onset_logits = onset_model.forward(mel_tensor)
        
    pitch_pred = pitch_logits.argmax(-1)
    onset_pred = onset_logits.argmax(-1)
        
    ### pitch_gt = pitch_gt.flatten().numpy().tolist()
    ### onset_gt = onset_gt.flatten().numpy().tolist()
    pitch_pred = pitch_pred.detach().cpu().flatten().numpy().tolist()
    onset_pred = onset_pred.detach().cpu().flatten().numpy().tolist()
    
    L = len(pitch_pred)
        
    for i in range(L):
        if onset_pred[i]:
            if pitch_pred[i] == 0:
                onset_pred[i] = 0
        else:
            if pitch_pred[i]:
                if i == 0 or pitch_pred[i-1] != pitch_pred[i]:
                    pitch_pred[i] = 0
                        
    # plot
        
    fig, axes = plt.subplots()
    plot(pitch_pred, onset_pred, axes, 'prediction')
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