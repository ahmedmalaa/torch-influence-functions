
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Code for influence functions computation in Pytorch
# ---------------------------------------------------------

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import torch
from torch.autograd import Variable 
import torch.nn.functional as nnf
from torch.utils.data import random_split
from torch.optim import SGD 
from torch.distributions import constraints
import torchvision as torchv
import torchvision.transforms as torchvt
from torch import nn

from influence_utils import *

def influence_function(model, 
                       train_index,
                       mode="stochastic",
                       batch_size=100, 
                       damp=1e-3, 
                       scale=1000, 
                       recursion_depth=1000):

    """
    Computes the influence function defined as H^-1 dLoss/d theta. This is the impact that each
    training data point has on the learned model parameters. 
    """
    
    if mode=="stochastic":

        IF = influence_stochastic_estimation(model, train_index, batch_size, damp, scale, recursion_depth)
    
    return IF    


def influence_stochastic_estimation(model, 
                                    train_index,
                                    batch_size=100, 
                                    damp=1e-3, 
                                    scale=1000, 
                                    recursion_depth=1000):

    """
    This function applies the stochastic estimation approach to evaluating influence function based on the power-series
    approximation of matrix inversion. Recall that the exact inverse Hessian H^-1 can be computed as follows:

    H^-1 = \sum^\infty_{i=0} (I - H) ^ j

    This series converges if all the eigen values of H are less than 1. 
    
    
    Arguments:
        loss: scalar/tensor, for example the output of the loss function
        rnn: the model for which the Hessian of the loss is evaluated 
        v: list of torch tensors, rnn.parameters(),
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    """
    
    SUBSAMPLES  = batch_size

    NUM_SAMPLES = model.X.shape[0]
    
    loss        = [model.loss_fn(model.y[train_index[_]], model.predict(model.X[train_index[_], :])) for _ in range(len(train_index))]
    
    grads       = [stack_torch_tensors(torch.autograd.grad(loss[_], model.parameters(), create_graph=True)) for _ in range(len(train_index))]
     
    IHVP_       = [grads[_].clone().detach() for _ in range(len(train_index))]

    
    for j in range(recursion_depth):
        
        sampled_indx = np.random.choice(list(range(NUM_SAMPLES)), SUBSAMPLES)[0]
        
        sampled_loss = model.loss_fn(model.y[sampled_indx], model.predict(model.X[sampled_indx, :]))
        
        IHVP_prev    = [IHVP_[_].clone().detach() for _ in range(len(train_index))]
        
        hvps_        = [stack_torch_tensors(hessian_vector_product(sampled_loss, model, [IHVP_prev[_]])) for _ in range(len(train_index))]
         
        IHVP_        = [g_ + (1 - damp) * ihvp_ - hvp_/scale for (g_, ihvp_, hvp_) in zip(grads, IHVP_prev, hvps_)] 
        
    return [IHVP_[_] / (scale * NUM_SAMPLES) for _ in range(len(train_index))] 
    
    
 