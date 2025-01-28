#!/bin/bash

PYTHON_SCRIPT="../src/reconstruction.py"

DATA_DIR="../data/"
G_OPTION="all"
M_OPTION="mlp_sd"
B_OPTION=5
S_OPTION=0
P_OPTION="../src/embedding/local_model/contrastive/mlp_${S_OPTION}/"
O_OPTION="../output/"


for i in {0..1000}
do
    start_time=$(date +%s)  # 记录开始时间
    python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -p $P_OPTION -s $S_OPTION -o $O_OPTION
    
    sleep 60  # 暂停 1 分钟
    
    # 计算运行时间
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    
    # 如果运行时间超过 10 分钟（600 秒），停止循环
    if [ $elapsed_time -ge 600 ]; then
        echo "Execution time exceeded 10 minutes. Stopping the loop."
        break
    fi
done
