"""Dataset for Graph Isomorphism Network(GIN)

Adapted from https://github.com/dmlc/dgl/blob/master/python/dgl/data/gindt.py to fit the input dimension of Ring-GNN.

"""

import os
import numpy as np

from utils import download, extract_archive, get_download_dir, _get_dgl_url
from dgl.graph import DGLGraph
from dgl.utils import Index
import torch as th

_url = 'https://raw.githubusercontent.com/weihua916/powerful-gnns/master/dataset.zip'


class GINDataset(object):
    """Datasets for Graph Isomorphism Network (GIN)
    Adapted from https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip.

    The dataset contains the compact format of popular graph kernel datasets, which includes: 
    MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, PROTEINS, PTC, REDDITBINARY, REDDITMULTI5K

    This datset class processes all data sets listed above. For more graph kernel datasets, 
    see :class:`TUDataset`

    Paramters
    ---------
    name: str
        dataset name, one of below -
        ('MUTAG', 'COLLAB', \
        'IMDBBINARY', 'IMDBMULTI', \
        'NCI1', 'PROTEINS', 'PTC', \
        'REDDITBINARY', 'REDDITMULTI5K')
    self_loop: boolean
        add self to self edge if true
    degree_as_nlabel: boolean
        take node degree as label and feature if true

    """

    def __init__(self, name, self_loop, device, degree_as_nlabel=False, line_graph=False):
        """Initialize the dataset."""

        self.name = name  # MUTAG
        self.ds_name = 'nig'
        self.extract_dir = self._download()
        self.file = self._file_path()

        self.device = device

        self.self_loop = self_loop
        self.line_graph = line_graph

        self.graphs = []
        self.labels = []

        # relabel
        self.glabel_dict = {}
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.N = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 0
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0

        # flags
        self.degree_as_nlabel = degree_as_nlabel
        self.nattrs_flag = False
        self.nlabels_flag = False
        self.verbosity = False

        # calc all values
        self._load()

        self._deg = []
        self._edges = []

        self.in_degrees = lambda g: g.in_degrees(
                Index(np.arange(0, g.number_of_nodes()))).unsqueeze(1).float()

        self._adj = []

        if self.line_graph:
            self._lgs = []
            self._lgs_deg = []
            self._lgs_edges = []
            self._pm = []
            self._pd = []
        
        self._preprocess()

        if self.degree_as_nlabel:
            self._deg_as_feature()
        elif self.name == 'PROTEINS' or self.name == 'PTC' or self.name == 'MUTAG':
            self._attr_as_feature()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        if self.line_graph:
            return self.graphs[idx], self.labels[idx], self._deg[idx], self._edges[idx], \
                self._lgs[idx], self._lgs_deg[idx], self._lgs_edges[idx], self._pm[idx], self._pd[idx]
        else:
            return self.graphs[idx], self.labels[idx], self._deg[idx], self._edges[idx], self._adj[idx]

    def _download(self):
        download_dir = get_download_dir()
        zip_file_path = os.path.join(
            download_dir, "{}.zip".format(self.ds_name))
        # TODO move to dgl host _get_dgl_url
        download(_url, path=zip_file_path)
        extract_dir = os.path.join(
            download_dir, "{}".format(self.ds_name))
        extract_archive(zip_file_path, extract_dir)
        return extract_dir

    def _file_path(self):
        return os.path.join(self.extract_dir, "dataset", self.name, "{}.txt".format(self.name))

    def _load(self):
        """ Loads input dataset from dataset/NAME/NAME.txt file

        """

        print('loading data...')
        with open(self.file, 'r') as f:
            # line_1 == N, total number of graphs
            self.N = int(f.readline().strip())

            for i in range(self.N):
                if (i + 1) % 10 == 0 and self.verbosity is True:
                    print('processing graph {}...'.format(i + 1))

                grow = f.readline().strip().split()
                # line_2 == [n_nodes, l] is equal to
                # [node number of a graph, class label of a graph]
                n_nodes, glabel = [int(w) for w in grow]

                # relabel graphs
                if glabel not in self.glabel_dict:
                    mapped = len(self.glabel_dict)
                    self.glabel_dict[glabel] = mapped

                self.labels.append(self.glabel_dict[glabel])

                g = DGLGraph()
                g.add_nodes(n_nodes)

                nlabels = []  # node labels
                nattrs = []  # node attributes if it has
                m_edges = 0

                for j in range(n_nodes):
                    nrow = f.readline().strip().split()

                    # handle edges and attributes(if has)
                    tmp = int(nrow[1]) + 2  # tmp == 2 + #edges
                    if tmp == len(nrow):
                        # no node attributes
                        nrow = [int(w) for w in nrow]
                        nattr = None
                    elif tmp > len(nrow):
                        nrow = [int(w) for w in nrow[:tmp]]
                        nattr = [float(w) for w in nrow[tmp:]]
                        nattrs.append(nattr)
                    else:
                        raise Exception('edge number is incorrect!')

                    # relabel nodes if it has labels
                    # if it doesn't have node labels, then every nrow[0]==0
                    if not nrow[0] in self.nlabel_dict:
                        mapped = len(self.nlabel_dict)
                        self.nlabel_dict[nrow[0]] = mapped

                    #nlabels.append(self.nlabel_dict[nrow[0]])
                    nlabels.append(nrow[0])

                    m_edges += nrow[1]
                    g.add_edges(j, nrow[2:])

                    # add self loop
                    if self.self_loop:
                        m_edges += 1
                        g.add_edge(j, j)

                    if (j + 1) % 10 == 0 and self.verbosity is True:
                        print(
                            'processing node {} of graph {}...'.format(
                                j + 1, i + 1))
                        print('this node has {} edgs.'.format(
                            nrow[1]))

                if nattrs != []:
                    nattrs = np.stack(nattrs)
                    g.ndata['attr'] = nattrs
                    self.nattrs_flag = True
                else:
                    nattrs = None

                g.ndata['label'] = np.array(nlabels)
                if len(self.nlabel_dict) > 1:
                    self.nlabels_flag = True

                assert len(g) == n_nodes

                # update statistics of graphs
                self.n += n_nodes
                self.m += m_edges

                self.graphs.append(g)

        # if no attr
        if not self.nattrs_flag:
            print('there are no node features in this dataset!')
            label2idx = {}
            # generate node attr by node degree
            if self.degree_as_nlabel:
                print('generate node features by node degree...')
                nlabel_set = set([])
                for g in self.graphs:
                    # actually this label shouldn't be updated
                    # in case users want to keep it
                    # but usually no features means no labels, fine.
                    g.ndata['label'] = g.in_degrees()
                    # extracting unique node labels
                    nlabel_set = nlabel_set.union(set(g.ndata['label'].numpy()))

                nlabel_set = list(nlabel_set)

                # in case the labels/degrees are not continuous number
                self.ndegree_dict = {
                    nlabel_set[i]: i
                    for i in range(len(nlabel_set))
                }
                label2idx = self.ndegree_dict
            # generate node attr by node label
            else:
                print('generate node features by node label...')
                label2idx = self.nlabel_dict

            for g in self.graphs:
                g.ndata['attr'] = np.zeros((
                    g.number_of_nodes(), len(label2idx)))
                g.ndata['attr'][range(g.number_of_nodes(
                )), [label2idx[nl.item()] for nl in g.ndata['label']]] = 1

        # after load, get the #classes and #dim
        self.gclasses = len(self.glabel_dict)
        self.nclasses = len(self.nlabel_dict)
        self.eclasses = len(self.elabel_dict)
        self.dim_nfeats = len(self.graphs[0].ndata['attr'][0])

        print('Done.')
        print(
            """
            -------- Data Statistics --------'
            #Graphs: %d
            #Graph Classes: %d
            #Nodes: %d
            #Node Classes: %d
            #Node Features Dim: %d
            #Edges: %d
            #Edge Classes: %d
            Avg. of #Nodes: %.2f
            Avg. of #Edges: %.2f
            Graph Relabeled: %s
            Node Relabeled: %s
            Degree Relabeled(If degree_as_nlabel=True): %s \n """ % (
                self.N, self.gclasses, self.n, self.nclasses,
                self.dim_nfeats, self.m, self.eclasses,
                self.n / self.N, self.m / self.N, self.glabel_dict,
                self.nlabel_dict, self.ndegree_dict))

    def _pm_pd(self, graphs):
        pm = [th.stack((g.all_edges()[0], th.arange(g.number_of_edges()))).to(self.device) for g in graphs]
        pd = [th.stack((g.all_edges()[1], th.arange(g.number_of_edges()))).to(self.device) for g in graphs]

        return pm, pd

    def _preprocess(self):
        self._deg = [self.in_degrees(g).to(self.device) for g in self.graphs]
        self._edges = [(g.all_edges()[0].to(self.device), g.all_edges()[1].to(self.device)) for g in self.graphs]
        self._adj = [g.adjacency_matrix().to_dense().to(self.device) for g in self.graphs]
        self._adj = [self._sym_normalize_adj(adj) for adj in self._adj]

        if self.line_graph:
            self._lgs = [g.line_graph(backtracking=False) for g in self.graphs]
            self._lgs_deg = [self.in_degrees(lg).to(self.device) for lg in self._lgs]
            self._lgs_edges = [(lg.all_edges()[0].to(self.device), lg.all_edges()[1].to(self.device)) for lg in self._lgs]
            self._pm, self._pd = self._pm_pd(self.graphs)
    
    def _sym_normalize_adj(self, adj):
        deg = th.sum(adj, dim = 0).squeeze()
        deg_inv = th.where(deg>0, 1./th.sqrt(deg), th.zeros(deg.size()).to(self.device))
        deg_inv = th.diag(deg_inv)
        return th.mm(deg_inv, th.mm(adj, deg_inv))

    def _deg_as_feature(self):
        for i, g in enumerate(self.graphs):
            new_adj = th.stack([self._adj[i] for j in range(self.dim_nfeats + 1)])
            for j, nl in enumerate(g.ndata['label']):
                new_adj[self.ndegree_dict[nl.item()] + 1][j][j] = 1
            
            self._adj[i] = new_adj
        
    def _attr_as_feature(self):
        for i, g in enumerate(self.graphs):
            new_adj = th.stack([self._adj[i] for j in range(self.nclasses + 1)])
            for j, nl in enumerate(g.ndata['label']):
                new_adj[self.nlabel_dict[nl]+1][j][j] = 1
            
            self._adj[i] = new_adj
