import numpy as np
import torch
from PQC import getHaar
import matplotlib.pyplot as plt

class QuantumState:
    def __init__(self,num=4):
        self.num = num
        self.state = np.zeros((2**num,2))
        self.state[0,0]=1
        self.state = torch.FloatTensor(self.state)
        self.state = torch.view_as_complex(self.state)
        # self.state.requires_grad_(True)
        # print(self.state)
    def SingleRX(self,theta):
        cost = torch.cos(theta/2)
        mjsint = -1j*torch.sin(theta/2)
        px = torch.cat((cost,mjsint,mjsint,cost))
        px = px.view((2,2))
        # print(px)
        return px
    def SingleRZ(self,theta):
        mexp = torch.exp(-0.5j*theta)
        pexp = torch.exp(0.5j*theta)
        zero = torch.zeros(1,)
        pz = torch.cat((mexp,zero,zero,pexp))
        return pz
    def SingleToMultiQubit(self,pos,p):
        eye = torch.eye(2)
        if pos==0:
            out = p
        else:
            out = eye
            for i in range(1,pos):
                out = torch.kron(out,eye)
            out = torch.kron(out,p)
        for i in range(pos+1,self.num):
            out = torch.kron(out,eye)
        out = torch.reshape(out,(2**self.num,-1))
        self.state = torch.matmul(out,self.state)
    def RX(self,pos,theta):
        self.SingleToMultiQubit(pos,self.SingleRX(theta))
    def RZ(self,pos,theta):
        self.SingleToMultiQubit(pos,self.SingleRZ(theta))