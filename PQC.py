import numpy as np
# import Layer
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import rv_continuous
from qiskit.circuit import Parameter, ParameterVector
from qiskit import *
from sklearn.metrics.cluster import adjusted_mutual_info_score as mi



class PQC:
    def __init__(self,name,num):
        self.backend = Aer.get_backend('statevector_simulator');
        self.circ = QuantumCircuit(num);
        self.name = name;
        self.seed = 14256;
        self.num = num;
        np.random.seed(self.seed);
        self.params = ParameterVector('Θ',0);
        # self.circ.rz(self.params[0],1);
        # self.circ.rz(self.params[1],1);

    def add(self,gate="rz",cc=0,c=0,o=0):
        pastlen = len(self.params.params);
        if gate == "rz" or gate == "rx":
            if o<0 or o>=self.num:
                print("Index Error");
                return;
            self.params.resize(pastlen+1);
            if gate == "rz":
                self.circ.rz(self.params[pastlen],o);
            if gate == "rx":
                self.circ.rx(self.params[pastlen],o);
        if gate == "crz" or gate == "crx":
            if o<0 or o>=self.num or c<0 or c>=self.num or o==c:
                print("Index Error");
                return;
            self.params.resize(pastlen+1);
            if gate == "crz":
                cir = QuantumCircuit(1, name="RZ");
                cir.rz(self.params[pastlen],0);
            if gate == "crx":
                cir = QuantumCircuit(1, name="RX");
                cir.rx(self.params[pastlen],0);
            temp = cir.to_gate().control(1);
            self.circ.append(temp,[c,o]);

    # def addLayer(self,num):


    def get_statevector(self):
        self.circ1 = self.circ.bind_parameters({self.params: np.random.uniform(0,2*np.pi,len(self.params.params))});
        result = execute(self.circ1,self.backend).result();
        out_state = result.get_statevector();
        self.statevector = np.asmatrix(out_state).T;
        return self.statevector;

    def draw(self):
        self.circ.draw('mpl');
        print(self.circ);

"""
Expressibility
"""

def Haar(F,N):
    if F<0 or F>1:
        return 0;
    return (N-1)*((1-F)**(N-2));


def kl_divergence(p, q):
    return np.sum(np.where(p*q != 0, p * np.log(p / q), 0));

class Haar_dist(rv_continuous):
    def _pdf(self,x,n):
        return Haar(x,2**n);


def expressibility(pqc, reps):
    arr = [];
    for i in range(reps):
        v1 = pqc.get_statevector().getH();
        v2 = pqc.get_statevector();
        fid = np.abs(v1*v2)**2;
        # print(v1,"&&",v2,"&&",np.abs(v1*v2),"&&",fid,"\n\n");
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
    # print(haar);
    # print(arr);
    # print(plt.hist(np.array(haar), bins=n_bins, alpha=0.5))
    # print(plt.hist(np.array(arr), bins=n_bins, alpha=0.5))
    # print(haar_pdf)
    # print(pqc_pdf);
    kl = kl_divergence(pqc_pdf,haar_pdf);
    plt.title("%s KL(P||Q) = %1.4f" % (pqc.name, kl))
    return kl;


"""
Entangling capability
"""

def I(b,j,n,vec):
    newvec = np.zeros((2**(n-1),1), dtype=complex);
    for new_index in range(2**(n-1)):
        original_index = new_index%(2**(n-j)) + (new_index//(2**(n-j)))*(2**(n-j+1)) + b*(2**(n-j));
        newvec[new_index]=vec[int(original_index)];
    return newvec;


def D(u,v,m):
    dist = 0;
    for i in range(m):
        for j in range(m):
            a = u[i]*v[j]-u[j]*v[i];
            # print(np.abs(a))
            dist += (1/2)*np.abs(a)**2;
    return dist;


def Q(n,vec):
    sum = 0;
    for j in range(n):
        sum += D(I(0,j+1,n,vec),I(1,j+1,n,vec),2**(n-1));
    return (sum * 4 / n)[0];


def entangling_capability(pqc, reps):
    sum = 0;
    for i in range(reps):
        sum += Q(pqc.num,pqc.get_statevector());
        if i%100==0 and i!=0:
            print(i,"\n");
    return sum/reps;

"""
unique-gate
"""

def unitary(circ,eta,phi,t):
    theta = np.arccos(-eta);
    circ.u3(theta,phi,t,0);

def V(circ,theta,phi,alpha,i):
    """
    theta: 0 ~ π
    phi: 0 ~ 2π
    alpha: 0 ~ 2π
    """
    circ.rz(-phi,i);
    circ.ry(-theta,i);
    circ.rz(alpha,i);
    circ.ry(theta,i);
    circ.rz(phi,i);