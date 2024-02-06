# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:05:29 2024

@author: jhowl
"""
import numpy as np
import torch
import pytorch_lightning as pl

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        return x.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        # None needed because of the optional arguments min_val and max v_val
        return grad_output.clone(), None, None

class AngularLoss(pl.LightningModule):
    """
    Evaluating mean angular error between the predicted and true event origins
    """
    def __init__(self, eps = .001): #gradients explode, so have to add eps
        super().__init__()
        self.high =1-eps
        self.low = -1+eps
        self.Clamp = DifferentiableClamp()
   
    def forward(self, y_pred, y_true):
        y_true = angles_to_unit_vector(y_true[:,0], y_true[:,1])
        y_pred = angles_to_unit_vector(y_pred[:,0], y_pred[:,1])
        
        scalar_prod = torch.sum(y_pred*y_true,dim = 1)
        scalar_prod = self.Clamp.apply(scalar_prod, self.low, self.high)
        return torch.mean(torch.abs(torch.arccos(scalar_prod)))

def angles_to_unit_vector(azimuth, zenith):
    """
    Transforms azimuth and zenith angles to unit vector in x,y,z coordinates
    """
    return torch.stack([
        torch.cos(azimuth) * torch.sin(zenith),
        torch.sin(azimuth) * torch.sin(zenith),
        torch.cos(zenith)
    ], dim=1)

def angular_dist_score(y_pred, y_true):
    """
    Competition metric used for validation
    """
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    az_true = y_true[:,0]
    az_pred = y_pred[:,0]
    zen_true = y_true[:,1]
    zen_pred = y_pred[:,1]
    #if not (np.all(np.isfinite(az_true)) and
    #        np.all(np.isfinite(zen_true)) and
    #        np.all(np.isfinite(az_pred)) and
    #        np.all(np.isfinite(zen_pred))):
    #    raise ValueError("All arguments must be finite")
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    scalar_prod =  np.clip(scalar_prod, -1, 1)
    return np.average(np.abs(np.arccos(scalar_prod)))