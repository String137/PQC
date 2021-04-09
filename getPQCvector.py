import numpy as np
import torch
import matplotlib.pyplot as plt
from QuantumState import *
def LayerInitializer(Layer,params):
    Layer.requires_grad_(False)
    for i in range(6):  #4
        Layer.weight[i] = params[6*i:6*(i+1)]
    for i in range(6):
        Layer.bias[i] = params[36+i]
def ParameterInitializer(params):
    for i in range(6):
        params[7*i] = 1
    for i in range(6):
        params[42+7*i] = 2*np.pi
    
def getPQCvector(params):
    Layer1 = torch.nn.Linear(6,6)
    LayerInitializer(Layer1,params[:42])
    Layer2 = torch.nn.Linear(6,6)
    LayerInitializer(Layer2,params[42:])
    input = torch.rand(6,)
    h1 = Layer1(input)
    a1 = torch.nn.Tanh()(h1)
    theta = Layer2(a1)
    # print("th",theta)
    q = QuantumState(2)
    q.RX(0,theta[0].view(1,))
    q.RX(1,theta[1].view(1,))
    q.RZ(0,theta[2].view(1,))
    q.RZ(1,theta[3].view(1,))
    q.CRX(0,1,theta[4].view(1,))
    q.CRX(1,0,theta[5].view(1,))
    v = q.state.view(-1,1)
    return v