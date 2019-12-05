python train_rebuttal.py \
--n-epochs 150 \
--lr 0.001 \
--iters-per-epoch 3 \
--output-folder results \
--dataset MUTAG \
--n-classes 2 \
--gpu 0 \
--batch-size 128 \
--test-batch-size 1 \
--fold-idx 0 \
--output-file mutag \
--nodeclasses 8 \
--avgnodenum 18

