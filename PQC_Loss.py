import numpy as np
import torch
import matplotlib.pyplot as plt
from getPQCvector import getPQCvector
from PQC import getHaar
# class GaussianHistogram(nn.Module):
#     def __init__(self, bins, min, max, sigma):
#         super(GaussianHistogram, self).__init__()
#         self.bins = bins
#         self.min = min
#         self.max = max
#         self.sigma = sigma
#         self.delta = float(max - min) / float(bins)
#         self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

#     def forward(self, x):
#         x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
#         x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
#         x = x.sum(dim=1)
#         return x
def histgauss(x,bins=75,sigma=0.1):
    delta = 1/float(bins)
    centers = delta * (torch.arange(bins).float()+0.5)
    x = torch.unsqueeze(x, 0) - torch.unsqueeze(centers, 1)
    x = torch.exp(-0.5*(x/sigma)**2) / (sigma * np.sqrt(np.pi*2)) * delta
    x = x.sum(dim=1)
    return x/x.sum(dim=0)
def histbin(x,locations = np.arange(0,1,1/75), radius=.1):
    locs = locations
    r = radius
    counts = []
        
    for loc in locs:
        dist = torch.abs(x - loc)
        # print(dist)
        ct = torch.relu(r - dist).sum(0) 
        counts.append(ct)
    
    out = torch.stack(counts, 0)
    out = out/torch.sum(out)
    return out

def Loss(params, verbose=0, bins=75, num=1000):
    Fidelity = None
    # print(Fidelity)
    haar_hist = torch.from_numpy(getHaar(reps=num,bins=bins,qubits=2))
    # params = torch.randn(2,requires_grad=True)
    for i in range(num):
        v1 = getPQCvector(params)
        v2 = getPQCvector(params)
        F = torch.abs(torch.matmul(torch.conj(torch.transpose(v1,0,1)),v2))**2
        if Fidelity is None:
            Fidelity = F.view(1,)
        else:
            Fidelity = torch.cat((Fidelity,F.view(1,)),-1)
    # print(Fidelity)
    # hist = histbin(Fidelity,locations=np.arange(0,1,1/bins),radius=2/bins)
    hist = histgauss(Fidelity,bins=250,sigma=1/bins)
    # print(hist)
    if verbose == 1:
        x = np.linspace(0,1,bins)
        plt.plot(x,torch.detach(hist))
        plt.plot(x,torch.detach(haar_hist))
        plt.figure()
    # kl = torch.kl_div(hist.log(),haar_hist,reduction=1)
    p = hist*(hist/haar_hist).log()
    p[p!=p]=0
    return p.sum()