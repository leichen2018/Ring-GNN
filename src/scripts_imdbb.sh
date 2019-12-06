python train.py \
--n-epochs 150 \
--lr 0.001 \
--iters-per-epoch 15 \
--output-folder results \
--dataset IMDBBINARY \
--n-classes 2 \
--gpu 0 \
--batch-size 128 \
--test-batch-size 1 \
--fold-idx 0 \
--output-file imdbb \
--avgnodenum 20 \
--nodeclasses 66 \
--degree_as_nlabels

