import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import rv_continuous
from qiskit.circuit import Parameter, ParameterVector
from qiskit import *
from sklearn.metrics.cluster import adjusted_mutual_info_score as mi



class PQC:
    def __init__(self,name,layer):
        self.backend = Aer.get_backend('statevector_simulator');
        self.circ = QuantumCircuit(layer);
        self.name = name;
        self.seed = 14256;
        self.layer = layer;
        np.random.seed(self.seed);
        if self.name == "rcz":
            self.theta = Parameter('θ');
            for index in range(layer):
                self.circ.h(index);
            c = QuantumCircuit(1,name="Rz");
            c.rz(self.theta,0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[0,1]);
        if self.name == "circ4":
            self.theta1 = ParameterVector('θ1', layer);
            self.theta2 = ParameterVector('θ2', layer);
            self.theta3 = ParameterVector('θ3', layer-1);
            # print(self.theta1)
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta3[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
        if self.name == "circ4c":
            self.theta1 = ParameterVector('θ1', layer);
            self.theta2 = ParameterVector('θ2', layer);
            self.theta3 = ParameterVector('θ3', layer);
            # print(self.theta1)
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta3[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta3[self.layer-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
        if self.name == "circ19":
            self.theta1 = ParameterVector('θ1', 3*layer);
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta1[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta1[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta1[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
        if self.name == "2circ19":
            self.theta1 = ParameterVector('θ1', 3*layer);
            self.theta2 = ParameterVector('θ2', 3*layer);
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta1[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta1[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta1[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta2[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta2[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta2[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);

        if self.name == "3circ19":
            self.theta1 = ParameterVector('θ1', 3*layer);
            self.theta2 = ParameterVector('θ2', 3*layer);
            self.theta3 = ParameterVector('θ3', 3*layer);
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta1[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta1[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta1[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta2[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta2[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta2[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta3[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta3[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta3[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta3[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);

        if self.name == "4circ19":
            self.theta1 = ParameterVector('θ1', 3*layer);
            self.theta2 = ParameterVector('θ2', 3*layer);
            self.theta3 = ParameterVector('θ3', 3*layer);
            self.theta4 = ParameterVector('θ4', 3*layer);
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta1[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta1[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta1[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta2[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta2[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta2[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta3[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta3[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta3[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta3[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta4[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta4[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta4[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta4[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
        
        if self.name == "5circ19":
            self.theta1 = ParameterVector('θ1', 3*layer);
            self.theta2 = ParameterVector('θ2', 3*layer);
            self.theta3 = ParameterVector('θ3', 3*layer);
            self.theta4 = ParameterVector('θ4', 3*layer);
            self.theta5 = ParameterVector('θ5', 3*layer);
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta1[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta1[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta1[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta2[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta2[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta2[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta3[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta3[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta3[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta3[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta4[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta4[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta4[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta4[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta5[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta5[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta5[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta5[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
        
        if self.name == "2circ19x":
            self.theta1 = ParameterVector('θ1', 3*layer);
            self.theta2 = ParameterVector('θ2', 3*layer);
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta1[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta1[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta1[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.x(i);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta2[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[self.layer+i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta2[self.layer*2+i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta2[self.layer*3-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);

        if self.name == "circ6":
            
            self.theta1 = ParameterVector('θ1', layer);
            self.theta2 = ParameterVector('θ2', layer);
            self.theta3 = ParameterVector('θ3', layer*(layer-1));
            self.theta4 = ParameterVector('θ4', layer);
            self.theta5 = ParameterVector('θ5', layer);
            
            # print(self.theta1)
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[i],i);
            for i in range(self.layer):
                for j in range(self.layer):
                    if i<j:
                        c = QuantumCircuit(1, name="Rx");
                        c.rx(self.theta3[(self.layer-1)*i+j-1],0);
                        temp = c.to_gate().control(1);
                        self.circ.append(temp,[i,j]);
                    elif i>j:
                        c = QuantumCircuit(1, name="Rx");
                        c.rx(self.theta3[(self.layer-1)*i+j],0);
                        temp = c.to_gate().control(1);
                        self.circ.append(temp,[i,j]);
            self.circ.barrier();
            for i in range(self.layer):
                self.circ.rx(self.theta4[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta5[i],i);


        if self.name == "circ19half":
            self.theta1 = ParameterVector('θ1', layer);
            self.theta2 = ParameterVector('θ2', layer);
            self.theta3 = ParameterVector('θ3', layer);
            # print(self.theta1)
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta3[i]/2,0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta3[self.layer-1]/2,0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta3[i]/2,0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta3[self.layer-1]/2,0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);

        if self.name == "circ4ca":
            self.theta1 = ParameterVector('θ1', layer);
            self.theta2 = ParameterVector('θ2', layer);
            self.theta3 = ParameterVector('θ3', layer);
            self.theta4 = ParameterVector('θ4', layer);
            self.theta5 = ParameterVector('θ5', layer);
            # print(self.theta1)
            for i in range(self.layer):
                self.circ.rx(self.theta1[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta2[i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1, name="Rx");
                c.rx(self.theta3[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1, name="Rx");
            c.rx(self.theta3[self.layer-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            for i in range(self.layer):
                self.circ.rx(self.theta4[i],i);
            for i in range(self.layer):
                self.circ.rz(self.theta5[i],i);

        if self.name == "V":
            self.theta = ParameterVector('θ',layer);
            self.phi = ParameterVector('φ',layer);
            self.alpha = ParameterVector('⍺',layer);
            for i in range(layer):
                self.circ.h(i);
            for i in range(layer):
                V(self.circ,self.theta[i],self.phi[i],self.alpha[i],i);
        if self.name == "V1":
            self.theta0 = ParameterVector('θ0',layer);
            self.theta1 = ParameterVector('θ1',layer);
            self.theta = ParameterVector('θ',layer);
            self.phi = ParameterVector('φ',layer);
            self.alpha = ParameterVector('⍺',layer);
            for i in range(layer):
                self.circ.rx(self.theta0[i],i);
            for i in range(layer):
                self.circ.rz(self.theta1[i],i);
            for i in range(layer):
                V(self.circ,self.theta[i],self.phi[i],self.alpha[i],i);
        if self.name == "circularcV":
            self.theta0 = ParameterVector('θ0',layer);
            self.theta1 = ParameterVector('θ1',layer);
            self.theta = ParameterVector('θ',layer);
            self.phi = ParameterVector('φ',layer);
            self.alpha = ParameterVector('⍺',layer);
            for i in range(layer):
                self.circ.rx(self.theta0[i],i);
            for i in range(layer):
                self.circ.rz(self.theta1[i],i);
            for i in range(layer-1):
                c = QuantumCircuit(1,name="V");
                V(c,self.theta[i],self.phi[i],self.alpha[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1,name="V");
            V(c,self.theta[layer-1],self.phi[layer-1],self.alpha[layer-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[layer-1,0]);
            
        if self.name == "new0":
            # U gate가 진짜 uniform한지 확인하기 위함입니다.
            self.y = ParameterVector('θ_i',layer);
            self.z = ParameterVector('ϕ_i',layer);
            self.eta = ParameterVector('η', layer);
            self.phi = ParameterVector('ϕ', layer);
            self.t = ParameterVector('t',layer);
            for i in range(self.layer):
                self.circ.ry(self.y[i],i);
                self.circ.rz(self.z[i],i);
            for i in range(self.layer):
                self.circ.u3(self.eta[i],self.phi[i],self.t[i],i);
        
        if self.name == "new1":
            self.eta = ParameterVector('η', layer-1);
            self.phi = ParameterVector('ϕ', layer-1);
            self.t = ParameterVector('t',layer-1);
            for i in range(self.layer):
                self.circ.h(i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1,name="U");             # 1 qubit짜리 회로를 생성합니다
                c.u3(self.eta[i],self.phi[i],self.t[i],0);  # U gate를 적용시킵니다.
                temp = c.to_gate().control(1);              # 윗 두 줄에서 만든 회로를 1개의 qubit이 control할 것이라고 말해줍니다
                self.circ.append(temp,[i,i+1]);             # 원래의 회로에 위치를 지정해서 추가합니다
        if self.name == "new2":
            self.eta1 = ParameterVector('η1', layer-1);
            self.phi1 = ParameterVector('ϕ1', layer-1);
            self.t1 = ParameterVector('t1',layer-1);
            self.eta2 = ParameterVector('η2', layer-1);
            self.phi2 = ParameterVector('ϕ2', layer-1);
            self.t2 = ParameterVector('t2',layer-1);
            for i in range(self.layer):
                self.circ.h(i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1,name="U");
                c.u3(self.eta1[i],self.phi1[i],self.t1[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            for i in range(self.layer-1):
                c = QuantumCircuit(1,name="U");
                c.u3(self.eta2[i],self.phi2[i],self.t2[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
        if self.name == "new3":
            self.eta = ParameterVector('η', layer);
            self.phi = ParameterVector('ϕ', layer);
            self.t = ParameterVector('t',layer);
            for i in range(self.layer):
                self.circ.h(i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1,name="U");
                c.u3(self.eta[i],self.phi[i],self.t[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1,name="U");
            c.u3(self.eta[self.layer-1],self.phi[self.layer-1],self.t[self.layer-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
        if self.name == "new4":
            self.eta1 = ParameterVector('η1', layer);
            self.phi1 = ParameterVector('ϕ1', layer);
            self.t1 = ParameterVector('t1',layer);
            self.eta2 = ParameterVector('η2', layer);
            self.phi2 = ParameterVector('ϕ2', layer);
            self.t2 = ParameterVector('t2',layer);
            for i in range(self.layer):
                self.circ.h(i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1,name="U");
                c.u3(self.eta1[i],self.phi1[i],self.t1[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1,name="U");
            c.u3(self.eta1[self.layer-1],self.phi1[self.layer-1],self.t1[self.layer-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);
            for i in range(self.layer-1):
                c = QuantumCircuit(1,name="U");
                c.u3(self.eta2[i],self.phi2[i],self.t2[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1,name="U");
            c.u3(self.eta2[self.layer-1],self.phi2[self.layer-1],self.t2[self.layer-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);

        if self.name == "new5":
            self.y = ParameterVector('θ_i',layer);
            self.z = ParameterVector('ϕ_i',layer);
            self.eta = ParameterVector('η', layer);
            self.phi = ParameterVector('ϕ', layer);
            self.t = ParameterVector('t',layer);
            for i in range(self.layer):
                self.circ.ry(self.y[i],i);
                self.circ.rz(self.z[i],i);
            for i in range(self.layer-1):
                c = QuantumCircuit(1,name="U");
                c.u3(self.eta[i],self.phi[i],self.t[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
            c = QuantumCircuit(1,name="U");
            c.u3(self.eta[self.layer-1],self.phi[self.layer-1],self.t[self.layer-1],0);
            temp = c.to_gate().control(1);
            self.circ.append(temp,[self.layer-1,0]);

        if self.name == "new6":
            self.eta1 = ParameterVector('η1', layer-1);
            self.phi1 = ParameterVector('ϕ1', layer-1);
            self.t1 = ParameterVector('t1',layer-1);
            for i in range(self.layer):
                self.circ.h(i);
            for i in range(self.layer-1):
                self.circ.cx(i,i+1);
            for i in range(self.layer-1):
                c = QuantumCircuit(1,name="U");
                c.u3(self.eta1[i],self.phi1[i],self.t1[i],0);
                temp = c.to_gate().control(1);
                self.circ.append(temp,[i,i+1]);
                

    def get(self):
        if self.name == "rcz":
            t = np.random.uniform(0,2*np.pi);
            # theta = Parameter('θ');
            self.circ1 = self.circ.assign_parameters({self.theta:t});
            result = execute(self.circ1,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name == "circ4":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ2 = self.circ1.bind_parameters({self.theta2: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.theta3: np.random.uniform(0,2*np.pi,self.layer-1)});
            result = execute(self.circ3,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name == "circ4c":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ2 = self.circ1.bind_parameters({self.theta2: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.theta3: np.random.uniform(0,2*np.pi,self.layer)});
            result = execute(self.circ3,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name == "circ19":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer*3)});
            result = execute(self.circ1,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name == "2circ19" or self.name == "2circ19x":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ2 = self.circ1.bind_parameters({self.theta2: np.random.uniform(0,2*np.pi,self.layer*3)});
            
            result = execute(self.circ2,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            # print(self.statevector);
            # print("\n\n\n\n*******\n\n\n\n");
            return self.statevector;
        if self.name == "3circ19":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ2 = self.circ1.bind_parameters({self.theta2: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ3 = self.circ2.bind_parameters({self.theta3: np.random.uniform(0,2*np.pi,self.layer*3)});
            
            result = execute(self.circ3,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name == "4circ19":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ2 = self.circ1.bind_parameters({self.theta2: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ3 = self.circ2.bind_parameters({self.theta3: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ4 = self.circ3.bind_parameters({self.theta4: np.random.uniform(0,2*np.pi,self.layer*3)});
            
            result = execute(self.circ4,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name == "5circ19":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ2 = self.circ1.bind_parameters({self.theta2: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ3 = self.circ2.bind_parameters({self.theta3: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ4 = self.circ3.bind_parameters({self.theta4: np.random.uniform(0,2*np.pi,self.layer*3)});
            self.circ5 = self.circ4.bind_parameters({self.theta5: np.random.uniform(0,2*np.pi,self.layer*3)});            
            result = execute(self.circ5,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name == "circ6":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ2 = self.circ1.bind_parameters({self.theta2: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.theta3: np.random.uniform(0,2*np.pi,self.layer*(self.layer-1))});
            self.circ4 = self.circ3.bind_parameters({self.theta4: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ5 = self.circ4.bind_parameters({self.theta5: np.random.uniform(0,2*np.pi,self.layer)});
            result = execute(self.circ5,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;

        if self.name == "circ19half":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ2 = self.circ1.bind_parameters({self.theta2: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.theta3: np.random.uniform(0,2*np.pi,self.layer)});
            result = execute(self.circ3,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name == "circ4ca":
            self.circ1 = self.circ.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ2 = self.circ1.bind_parameters({self.theta2: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.theta3: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ4 = self.circ3.bind_parameters({self.theta4: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ5 = self.circ4.bind_parameters({self.theta5: np.random.uniform(0,2*np.pi,self.layer)});
            result = execute(self.circ5,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;

        if self.name == "V":
            self.circ1 = self.circ.bind_parameters({self.theta: np.random.uniform(0,np.pi,self.layer)});
            self.circ2 = self.circ1.bind_parameters({self.phi: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.alpha: np.random.uniform(0,2*np.pi,self.layer)});
            result = execute(self.circ3,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;

        if self.name =="V1":
            self.circ1 = self.circ.bind_parameters({self.theta0: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ2 = self.circ1.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.theta: np.random.uniform(0,np.pi,self.layer)});
            self.circ4 = self.circ3.bind_parameters({self.phi: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ5 = self.circ4.bind_parameters({self.alpha: np.random.uniform(0,2*np.pi,self.layer)});
            result = execute(self.circ5,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;

        if self.name =="circularcV":
            self.circ1 = self.circ.bind_parameters({self.theta0: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ2 = self.circ1.bind_parameters({self.theta1: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.theta: np.random.uniform(0,np.pi,self.layer)});
            self.circ4 = self.circ3.bind_parameters({self.phi: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ5 = self.circ4.bind_parameters({self.alpha: np.random.uniform(0,2*np.pi,self.layer)});
            result = execute(self.circ5,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;

        if self.name == "new0":
            self.circ1 = self.circ.bind_parameters({self.y: np.arccos(-np.random.uniform(-1,1,self.layer))});
            self.circ2 = self.circ1.bind_parameters({self.z: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.eta: np.arccos(-np.random.uniform(-1,1,self.layer))});
            self.circ4 = self.circ3.bind_parameters({self.phi: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ5 = self.circ4.bind_parameters({self.t: np.random.uniform(0,2*np.pi,self.layer)});
            
            result = execute(self.circ5,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;

        if self.name =="new1":
            # parameter에 랜덤한 값을 샘플링해서 할당해줍니다
            # eta의 경우엔, 원래 theta = arccos(-eta)가 들어가야하는데 위에서 잘 해결되지 않아서 여기서 변환해줬습니다.
            self.circ1 = self.circ.bind_parameters({self.eta: np.arccos(-np.random.uniform(-1,1,self.layer-1))});
            self.circ2 = self.circ1.bind_parameters({self.phi: np.random.uniform(0,2*np.pi,self.layer-1)});
            self.circ3 = self.circ2.bind_parameters({self.t: np.random.uniform(0,2*np.pi,self.layer-1)});
            # State vector를 얻으면 됩니다!
            result = execute(self.circ3,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name =="new2":
            self.circ1 = self.circ.bind_parameters({self.eta1: np.arccos(-np.random.uniform(-1,1,self.layer-1))});
            self.circ2 = self.circ1.bind_parameters({self.phi1: np.random.uniform(0,2*np.pi,self.layer-1)});
            self.circ3 = self.circ2.bind_parameters({self.t1: np.random.uniform(0,2*np.pi,self.layer-1)});
            self.circ4 = self.circ3.bind_parameters({self.eta2: np.arccos(-np.random.uniform(-1,1,self.layer-1))});
            self.circ5 = self.circ4.bind_parameters({self.phi2: np.random.uniform(0,2*np.pi,self.layer-1)});
            self.circ6 = self.circ5.bind_parameters({self.t2: np.random.uniform(0,2*np.pi,self.layer-1)});
            result = execute(self.circ6,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name =="new3":
            self.circ1 = self.circ.bind_parameters({self.eta: np.arccos(-np.random.uniform(-1,1,self.layer))});
            self.circ2 = self.circ1.bind_parameters({self.phi: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.t: np.random.uniform(0,2*np.pi,self.layer)});
            result = execute(self.circ3,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name =="new4":
            self.circ1 = self.circ.bind_parameters({self.eta1: np.arccos(-np.random.uniform(-1,1,self.layer))});
            print(self.eta1)
            self.circ2 = self.circ1.bind_parameters({self.phi1: np.random.uniform(0,2*np.pi,self.layer)});
            print(self.phi1)
            self.circ3 = self.circ2.bind_parameters({self.t1: np.random.uniform(0,2*np.pi,self.layer)});
            print(self.t1)
            self.circ4 = self.circ3.bind_parameters({self.eta2: np.arccos(-np.random.uniform(-1,1,self.layer))});
            print(self.eta2)
            self.circ5 = self.circ4.bind_parameters({self.phi2: np.random.uniform(0,2*np.pi,self.layer)});
            print(self.phi2)
            self.circ6 = self.circ5.bind_parameters({self.t2: np.random.uniform(0,2*np.pi,self.layer)});
            print(self.t2)
            result = execute(self.circ6,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;
        if self.name == "new5":
            self.circ1 = self.circ.bind_parameters({self.y: np.arccos(-np.random.uniform(-1,1,self.layer))});
            self.circ2 = self.circ1.bind_parameters({self.z: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ3 = self.circ2.bind_parameters({self.eta: np.arccos(-np.random.uniform(-1,1,self.layer))});
            self.circ4 = self.circ3.bind_parameters({self.phi: np.random.uniform(0,2*np.pi,self.layer)});
            self.circ5 = self.circ4.bind_parameters({self.t: np.random.uniform(0,2*np.pi,self.layer)});
            result = execute(self.circ5,self.backend).result();
            out_state = result.get_statevector();
            self.statevector = np.asmatrix(out_state).T;
            return self.statevector;

        if self.name =="new6":
            self.circ1 = self.circ.bind_parameters({self.eta1: np.arccos(-np.random.uniform(-1,1,self.layer-1))});
            self.circ2 = self.circ1.bind_parameters({self.phi1: np.random.uniform(0,2*np.pi,self.layer-1)});
            self.circ3 = self.circ2.bind_parameters({self.t1: np.random.uniform(0,2*np.pi,self.layer-1)});
            result = execute(self.circ3,self.backend).result();
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
        v1 = pqc.get().getH();
        v2 = pqc.get();
        fid = np.abs(v1*v2)**2;
        # print(v1,"&&",v2,"&&",np.abs(v1*v2),"&&",fid,"\n\n");
        arr.append(fid[0,0]);
        if i%100==0 and i!=0:
            print(i,"\n");
    haar = [];
    h = Haar_dist(a=0,b=1,name="haar");
    for i in range(reps):
        haar.append(h.ppf((i+1)/reps,pqc.layer));
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
        sum += Q(pqc.layer,pqc.get());
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