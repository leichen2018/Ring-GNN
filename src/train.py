from __future__ import division
import time
from datetime import datetime as dt

import argparse
from itertools import permutations

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

from model import G_invariant

from gindt import GINDataset

import sys

from utils import output_csv

test_acc_list = []

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

criterion = nn.CrossEntropyLoss()

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph[1] for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    start_idx = 0

    rand_seq = np.random.permutation(len(train_graphs))

    for pos in pbar:
        if start_idx + args.batch_size <= len(train_graphs):
            selected_idx = rand_seq[start_idx:start_idx + args.batch_size]
        else:
            selected_idx = np.concatenate((rand_seq[start_idx:], rand_seq[:start_idx + args.batch_size - len(train_graphs)]))
        
        start_idx = (start_idx + args.batch_size) % len(train_graphs)

        batch_graph = [train_graphs[idx] for idx in selected_idx]

        adj = [graph[4] for graph in batch_graph]

        optimizer.zero_grad()
        for i in range(args.batch_size):
            #adj_0 = adj[i].unsqueeze(0)
            if args.nodeclasses == 1:
                adj_0 = adj[i].unsqueeze(0).unsqueeze(0)
            else:
                adj_0 = adj[i].unsqueeze(0)
            output = model(adj_0)
            labels = th.LongTensor([batch_graph[i][1]]).to(device)
            #labels = th.LongTensor([graph[1] for graph in batch_graph]).to(device)
            loss = criterion(output, labels)/args.batch_size

            loss.backward()
            loss_accum += loss.detach().cpu().numpy()
            
        #output = model(adj)

        #labels = th.LongTensor([graph[1] for graph in batch_graph]).to(device)

        #compute loss
        #loss = criterion(output, labels)

        #backprop
        #optimizer.zero_grad()
        #loss.backward()         
        optimizer.step()
        
        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

def pass_data_iteratively(args, model, graphs, device, minibatch_size = 1):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch_graph = [graphs[idx] for idx in sampled_idx]

        adj = [graph[4] for graph in batch_graph]
        if args.nodeclasses == 1:
            adj_0 = adj[0].unsqueeze(0).unsqueeze(0)
        else:
            adj_0 = adj[0].unsqueeze(0)
        #adj = th.stack([adj[0], adj[0]+th.diag(th.ones(adj[0].size()[0])).to(device), adj[0], adj[0]]).unsqueeze(0)

        output.append(model(adj_0).detach())
    return th.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(args, model, train_graphs, device, args.test_batch_size)
    pred = output.max(1, keepdim=True)[1]
    labels = th.LongTensor([graph[1] for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(args, model, test_graphs, device, args.test_batch_size)

    pred = output.max(1, keepdim=True)[1]

    labels = th.LongTensor([graph[1] for graph in test_graphs]).to(device)
    
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    test_acc_list.append(acc_test)

    return acc_train, acc_test

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size', default=32)
    parser.add_argument('--gpu', type=int, help='GPU index', default=-1)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.01)
    parser.add_argument('--n-epochs', type=int, help='Number of epochs', default=1)
    parser.add_argument('--n-hidden', type=int, help='Number of features', default=64)
    parser.add_argument('--n-layers', type=int, help='Number of layers', default=5)
    parser.add_argument('--optim', type=str, help='Optimizer', default='Adam')
    parser.add_argument('--save-path', type=str, default='model.pkl')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--iters-per-epoch', type=int, default=50)
    parser.add_argument('--dataset', type=str, default="IMDBBINARY")
    parser.add_argument('--fold-idx', type=int, default=0)
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--n-input', type=int, default=1)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--output-file', type=str, default='output')
    parser.add_argument('--output-folder', type=str, default='results')
    parser.add_argument('--ops', type=int, default=1)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--nodeclasses', type=int, default=1)
    parser.add_argument('--avgnodenum', type=int, default=10)
    parser.add_argument('--degree_as_nlabels', action="store_true")
    args = parser.parse_args()

    dev0 = th.device('cpu')
    dev = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

    feats = [args.n_input] + [args.n_hidden] * (args.n_layers-1) + [args.n_classes]
    model = G_invariant(args.nodeclasses, args.n_classes, avgnodenum = args.avgnodenum, hidden = 32, radius = args.radius).to(dev)

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    th.manual_seed(1)
    np.random.seed(1) 

    dataset = GINDataset(args.dataset, self_loop = False, device = dev, degree_as_nlabel=args.degree_as_nlabels)

    train_graphs, test_graphs = separate_data(dataset, args.seed, args.fold_idx)

    for i in range(args.n_epochs):
        scheduler.step()
        train(args, model, dev, train_graphs, optimizer, i)
        test(args, model, dev, train_graphs, test_graphs, i)
    
    output_csv(args.output_folder + '/0729_' + args.output_file + '_' + str(args.fold_idx) + '.csv', test_acc_list)

if __name__ == '__main__':
    main()
