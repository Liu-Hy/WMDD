#!/bin/bash

RUN_PRETRAIN=false  # Set to false to skip pretraining
DATASET="imagenette"
MODEL="resnet18"
EVAL_MODEL="resnet18"
DATA_ROOT="/home/techt/Desktop/"
EXPERIMENT=23
MODEL_EXP=23
ITERATION=2000
KD_EPOCHS=300
DEBUG=false
IPC=10
GPU=0
R_WB=1
WB=false
PER_CLASS_BN=false

# Parse command-line arguments. All flags are optional.
# Usage: bash run.sh -x 2 -y 1 -d imagenette -u 0 -c 10 -r /home/user/data/ -n -w -b 3.0
# -x is the experiment id. Arguments and results will be saved in ./log/{experiment}.json
# If -p is included, it pretrains a model from scratch and saves it with the id given by '-y'.
# -y is the id of the teacher model under the (dataset, model) category. Make sure the model exists of '-p' is not set

while getopts ":pd:m:e:x:y:r:i:z:gc:u:wnb:a:t:" opt; do
  case $opt in
    p) RUN_PRETRAIN=true;;
    d) DATASET="$OPTARG";;
    m) MODEL="$OPTARG";;
    e) EVAL_MODEL="$OPTARG";;
    x) EXPERIMENT="$OPTARG";;
    y) MODEL_EXP="$OPTARG";;
    r) DATA_ROOT="$OPTARG";;
    i) ITERATION="$OPTARG";;
    z) KD_EPOCHS="$OPTARG";;
    g) DEBUG=true;;
    c) IPC="$OPTARG";;
    u) GPU="$OPTARG";;
    w) WB=true;;
    n) PER_CLASS_BN=true;;
    b) R_BN="$OPTARG";;
    a) LR="$OPTARG";;
    t) R_WB="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# Test if the code can run
if [ "$DEBUG" = true ]; then
    ITERATION=2
    KD_EPOCHS=2
    IPC=2
fi

if [[ "${DATASET}" == "tiny-imagenet" ]]; then
    # Only set R_BN and LR if they weren't provided as command-line arguments
    : ${R_BN:=1.0}
    : ${LR:=0.1}
else
    : ${R_BN:=0.01}
    : ${LR:=0.25}
fi

# Handle the case when the dataset has 10 classes and IPC is small, the effective batchsize in the relableling step is
# smaller, causing error in the FKD step.
if [[ "${DATASET}" != "tiny-imagenet" && "${DATASET}" != "imagenet" && "${IPC}" -lt 10 ]]; then
    KD_BATCH_SIZE=$((10 * IPC))
else
    KD_BATCH_SIZE=50
fi

DATA_PATH="${DATA_ROOT}${DATASET}"



cd ./train/
CUDA_VISIBLE_DEVICES=${GPU} python train_FKD.py \
    --dataset ${DATASET} \
    --model ${EVAL_MODEL} \
    --batch-size ${KD_BATCH_SIZE} \
    --epochs ${KD_EPOCHS} \
    --exp-name ${EXPERIMENT} \
    --cos \
    --temperature 20 \
    --workers 8 \
    --gradient-accumulation-steps 1 \
    --train-dir "../recover/syn_data/${DATASET}/${EXPERIMENT}" \
    --val-dir ${DATA_PATH}/val \
    --fkd-path "../relabel/FKD_cutmix_fp16/${DATASET}/${EXPERIMENT}" \
    --mix-type 'cutmix' \
    --output-dir "./save/final_rn18_fkd/${EXPERIMENT}/"
cd ..
