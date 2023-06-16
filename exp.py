from itertools import product
import subprocess

OPT_NAMES = ('Adam', 'SGD')
LEARNING_RATES = (1e-4, )


for opt_name, lr in product(OPT_NAMES, LEARNING_RATES):
    subprocess.call(f'python train.py \
                    --opt_name {opt_name} \
                    --lr {lr}', \
                    shell=True
                )