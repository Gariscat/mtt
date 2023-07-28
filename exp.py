from itertools import product
import subprocess

PROJECT_NAME = 'MTTLeadAdamW'
# SIZES = ([512, 512], [256, 256],)

"""
for opt_name, lr, ext_name in product(OPT_NAMES, LEARNING_RATES, EXTRACTOR_NAMES):
    subprocess.call(f'python train.py \
                    --opt_name {opt_name} \
                    --lr {lr} \
                    --extractor_name {ext_name}', \
                    shell=True
                )
"""    
for rnn_type, num_layers in (
    # transformers
    (None, 6),
    (None, 3),
    # rnn
    ('lstm', 6),
    ('lstm', 3),
    ('gru', 6),
    ('gru', 3),
):
    rnn_type = f'--rnn_type {rnn_type}' if rnn_type else ''
    subprocess.call(f'python train.py \
                    {rnn_type} \
                    --num_layers {num_layers} \
                    --project_name {PROJECT_NAME}', \
                    shell=True
                )