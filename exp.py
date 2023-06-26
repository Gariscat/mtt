from itertools import product
import subprocess

OPT_NAMES = ('SGD',)
LEARNING_RATES = (1e-4, )
EXTRACTOR_NAMES = (None, )
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
for h in (256, 512, 1024):
    subprocess.call(f'python train.py \
                    --opt_name SGD \
                    --lr {1e-4} \
                    --extractor_name MLP \
                    --hidden_size {h}', \
                    shell=True
                )