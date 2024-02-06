# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:42:49 2024

@author: jhowl
"""
import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from Layers import DynEdgeConv, calculate_global_variables
from Loss import AngularLoss, angular_dist_score

GLOBAL_POOLINGS = {
    "max": global_max_pool,
    "add": global_add_pool,
    "mean": global_mean_pool,
}

class EdgeModel(torch.nn.Module):
    """
    Edge Convolution Model
    """
    def __init__(self, num_features):
        """
        Construct model:
        Args:
            num_features: number of features in dataset
        """
        super().__init__()
        self.n_features = num_features
        self.n_outputs = 2
        self.convolution = nn.ModuleList([DynEdgeConv(nn.Sequential(Linear(2*self.n_features, 128), nn.LeakyReLU(),
                                             Linear(128, 256), nn.LeakyReLU())),
                                         DynEdgeConv(nn.Sequential(Linear(512, 336), nn.LeakyReLU(),
                                             Linear(336, 256), nn.LeakyReLU())),
                                         DynEdgeConv(nn.Sequential(Linear(512, 336), nn.LeakyReLU(),
                                             Linear(336, 256), nn.LeakyReLU()))])
        self.post_process = nn.Sequential(Linear(774, 336), nn.LeakyReLU(),
                                          Linear(336, 256), nn.LeakyReLU())
        self.readout = nn.Sequential(Linear(778, 128), nn.LeakyReLU())
        self.out = nn.Sequential(Linear(128, self.n_outputs),
                                      BatchNorm1d(self.n_outputs), nn.LeakyReLU())

    def forward(self, x, edge_index, batch):
        """
        Forward pass
        """
        # 1. Obtain node embeddings
        global_features = calculate_global_variables(x, 
                                                     edge_index,
                                                     batch,
                                                    )

        graphs = [x]
        for i, layer in enumerate(self.convolution):
            graph, edge_index = layer(graphs[i], edge_index)
            graphs.append(graph)
        x = torch.cat(graphs, dim=1)

        x = self.post_process(x)

        pool_x = []
        for pool in GLOBAL_POOLINGS.values():
            pool_x.append(pool(x, batch))

        pool_x.append(global_features)
        pool_x = torch.cat(pool_x, dim=1)

        x = self.readout(pool_x)
        x = self.out(x)
        return x
    
    def fit(self, loader, epochs=5, device='cuda'):
        """
        Model training
        """
        loss_fn = AngularLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  
        self.train()

        for epoch in range(epochs+1):
            epoch_loss = 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(out, batch.y.reshape(out.size()))
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()   
            epoch_loss = epoch_loss / len(loader)
            print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}')

    @torch.no_grad()
    def validate(self, loader, device='cuda'):
        """
        Model validation
        """
        loss_fn = angular_dist_score
        epoch_loss = 0
        predictions = []

        self.eval()
        for batch in loader:
            batch = batch.to(device)
            out = self(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y.reshape(out.size()))
            epoch_loss += loss.item()
            predictions.append(out)

        epoch_loss = epoch_loss / len(loader)
        print(f'Loss: {epoch_loss:.4f}')
        return torch.cat(predictions, 0)
    
    @torch.no_grad()
    def test(self, loader, device='cuda'):
        """
        Model inference
        """
        predictions = []

        self.eval()
        for batch in loader:
            batch.to(device)
            out = self(batch.x, batch.edge_index, batch.batch)
            predictions.append(out)
        return torch.cat(predictions, 0)