# Ring-GNN

Source code for `On the equivalence between graph isomorphism testing and function approximation with GNNs` accepted by NeurIPS 2019.

Authors: Zhengdao Chen, [Soledad Villar](https://cims.nyu.edu/~villar/), [Lei Chen](https://leichen2018.github.io), [Joan Bruna](https://cims.nyu.edu/~bruna/).

## Environment

PyTorch, DGL

## Directory Initialization

```
mkdir results
```

## Running Scripts

with GPU:
```
bash scripts_*.sh
```

## Acknowledgement

* `gindt.py` is adapted from [https://github.com/dmlc/dgl/blob/master/python/dgl/data/gindt.py](https://github.com/dmlc/dgl/blob/master/python/dgl/data/gindt.py)

* `model.py` is based on a PyTorch extension of [https://github.com/Haggaim/InvariantGraphNetworks/blob/master/layers/equivariant_linear.py](https://github.com/Haggaim/InvariantGraphNetworks/blob/master/layers/equivariant_linear.py) and [https://github.com/Haggaim/InvariantGraphNetworks/blob/master/models/invariant_basic.py](https://github.com/Haggaim/InvariantGraphNetworks/blob/master/models/invariant_basic.py). The original version is in TensorFlow.

* Part of `train.py` is from [https://github.com/weihua916/powerful-gnns/blob/master/main.py](https://github.com/weihua916/powerful-gnns/blob/master/main.py).