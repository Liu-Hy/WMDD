# wandb disabled
wandb disabled
wandb online

python train_FKD.py \
    --wandb-project 'final_rn18_fkd' \
    --dataset imagenette \
    --batch-size 100 \
    --model resnet18 \
    --cos \
    -j 4 --gradient-accumulation-steps 2 \
    -T 20 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn18_fkd/rn18_[4K]_T20/ \
    --train-dir ../recover/syn_data/imagenette/rn18_bn0.01_[4K]_x_l2_x_tv.crop_2 \
    --val-dir /media/techt/DATA/data/imagenette/val \
    --fkd-path ../relabel/FKD_cutmix_fp16/imagenette/