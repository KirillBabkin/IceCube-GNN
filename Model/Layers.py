# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:44:00 2024

@author: jhowl
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils.homophily import homophily

from torch_scatter import scatter_mean

def calculate_xyzt_homophily(x, edge_index, batch):
    """
    Calculate xyzt-homophily from a batch of graphs.

    Homophily is a graph scalar quantity that measures the likeness of
    variables in nodes. Notice that this calculator assumes a special order of
    input features in x.

    Returns:
        Tuple, each element with shape [batch_size,1].
    """
    hx = homophily(edge_index, x[:, 0], batch).reshape(-1, 1)
    hy = homophily(edge_index, x[:, 1], batch).reshape(-1, 1)
    hz = homophily(edge_index, x[:, 2], batch).reshape(-1, 1)
    ht = homophily(edge_index, x[:, 3], batch).reshape(-1, 1)
    return hx, hy, hz, ht

def calculate_global_variables(x, edge_index, batch):
    """
    Calculate global variables.
    """
    # Calculate homophily (scalar variables)
    h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

    # Calculate mean features
    global_means = scatter_mean(x, batch, dim=0)

    # Add global variables
    global_variables = torch.cat(
        [
            global_means,
            h_x,
            h_y,
            h_z,
            h_t,
        ],
        dim=1,
    )

    return global_variables


class DynEdgeConv(EdgeConv, pl.LightningModule):
    """
    Dynamical edge convolution layer.
    """

    def __init__(self, nn, aggr = "max", nb_neighbors = 8, features_subset = None, **kwargs,):
        """
        Construct `DynEdgeConv`.
        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv`.
            aggr: Aggregation method to be used with `EdgeConv`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            **kwargs: Additional features to be passed to `EdgeConv`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        # Base class constructor
        super().__init__(nn=nn, aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset

    def forward(self, x, edge_index, batch = None):
        """
        Forward pass.
        """
        # Standard EdgeConv forward pass
        x = super().forward(x, edge_index)

        # Recompute adjacency
        edge_index = knn_graph(
            x=x[:, self.features_subset],
            k=self.nb_neighbors,
            batch=batch,
        ).to('cuda')

        return x, edge_index