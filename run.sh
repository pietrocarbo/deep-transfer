#!/usr/bin/env bash

source /home/pietro/anaconda3/bin/activate deep-transfer

python3 main.py --contentDir 'pytorch_tutorials/5_transferlearn/hymenoptera_data/train/ants' --styleDir 'pytorch_tutorials/5_transferlearn/hymenoptera_data/train/bees'