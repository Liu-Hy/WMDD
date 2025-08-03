python pretrain.py \
--dataset imagenette \
--model resnet18 \
--opt sgd --lr 0.025 \
-wd 1e-4 -b 32 --lr-scheduler cosine --epochs 50 \
--augmix-severity 0 \
--ra-magnitude 0 \
--data-path /media/techt/DATA/data/imagenette

