import numpy as np
import torch
import matplotlib.pyplot as plt
from QuantumState import *
def LayerInitializer(Layer,params):
    Layer.requires_grad_(False)
    for i in range(2):  #4
        Layer.weight[i] = params[2*i:2*(i+1)]
    for i in range(2):
        Layer.bias[i] = params[4+i]
def getPQCvector(params):
    Layer1 = torch.nn.Linear(2,2)
    LayerInitializer(Layer1,params[:6])
    Layer2 = torch.nn.Linear(2,2)
    LayerInitializer(Layer2,params[6:])
    input = torch.rand(2,)
    h1 = Layer1(input)
    a1 = torch.nn.Tanh()(h1)
    theta = Layer2(a1)
    # print("th",theta)
    q = QuantumState(2)
    q.RX(0,theta[0].view(1,))
    q.RX(1,theta[1].view(1,))
    v = q.state.view(-1,1)
    return v