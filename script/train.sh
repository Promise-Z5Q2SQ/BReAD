#!/bin/bash

PYTHON_SCRIPT="../src/train.py"

DATA_DIR="../data/"
G_OPTION="all"
M_OPTION="mlp_sd"
S_OPTION=8
P_OPTION="mlpsd_s${S_OPTION}_tmp.pth"
O_OPTION="../src/embedding/local_model/contrastive/"

python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -s $S_OPTION -o $O_OPTION --debug
# python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -p $P_OPTION -s $S_OPTION -o $O_OPTION --debug

# for i in {0..15}
# do
#     python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -s $i -o $O_OPTION --debug
#     # P_OPTION1="eegnet_s${i}_1x_1.pth"
#     # python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -p $P_OPTION -s $S_OPTION -o $O_OPTION --debug
# done