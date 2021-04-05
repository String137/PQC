import torch
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import rv_continuous
from qiskit.circuit import Parameter, ParameterVector
from qiskit import *
from sklearn.metrics.cluster import adjusted_mutual_info_score as mi
from PQC import *

class NNPQC(torch.nn.Module):
    def __init__(self, D_in=12, H=12, D_out=12):
        
        super(NNPQC, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)


    def expressibility(pqc, reps):
        arr = [];
        for i in range(reps):
            v1 = pqc.get(2*np.pi*np.random.uniform(size=(12,))).getH();
            v2 = pqc.get(2*np.pi*np.random.uniform(size=(12,)));
            fid = np.abs(v1*v2)**2;
            arr.append(fid[0,0]);
            if i%100==0 and i!=0:
                print(i,"\n");
        haar = [];
        h = Haar_dist(a=0,b=1,name="haar");
        for i in range(reps):
            haar.append(h.ppf((i+1)/reps,pqc.num));
        n_bins = 75;
        haar_pdf = plt.hist(np.array(haar), bins=n_bins, alpha=0.5,range=(0,1))[0]/reps; 
        pqc_pdf = plt.hist(np.array(arr), bins=n_bins, alpha=0.5, range=(0,1))[0]/reps;
        kl = kl_divergence(pqc_pdf,haar_pdf);
        plt.title("%s KL(P||Q) = %1.4f" % (pqc.name, kl))
        return kl;

    def forward(self, x):
        
        h_relu = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu).clamp(min=0)
        y_pred = self.linear3(h_relu2)

        return y_pred