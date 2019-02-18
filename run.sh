#!/bin/bash

python train.py character_models_strindberg 2 0 5 32;
python train.py character_models_strindberg 2 1 5 256;
python train.py character_models_strindberg 2 2 5 1024;
python train.py character_models_strindberg 3 0 5 32;
python train.py character_models_strindberg 3 1 5 256;
# python train.py character_models_strindberg 3 2 5 1024;

aws ec2 stop-instances --instance-ids i-03630fe8cd14a4caa;
