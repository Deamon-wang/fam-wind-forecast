from pyvqnet.qnn.measure import ProbsMeasure
# pyvqnet >= 2.18 将 torch 接口移至 qnn.vqc.sv.torch，旧版为 qnn.vqc.torch，这里做兼容
try:
    from pyvqnet.qnn.vqc.torch import QMachine,QModule,MeasureAll
    from pyvqnet.qnn.vqc.torch import QMachine, rx,ry,rz,cnot,RX, RY,RZ, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval,MeasureAll,QModule
    from pyvqnet.qnn.vqc.torch import TorchQpanda3QuantumLayer
except ImportError:  # pragma: no cover
    from pyvqnet.qnn.vqc.sv.torch import QMachine,QModule,MeasureAll
    from pyvqnet.qnn.vqc.sv.torch import QMachine, rx,ry,rz,cnot,RX, RY,RZ, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval,MeasureAll,QModule
    from pyvqnet.qnn.vqc.sv.torch import TorchQpanda3QuantumLayer
from pyvqnet.nn import Parameter

import numpy as np

# class Angle_VQC_cir(QModule):
#     def __init__(self, n_bit, qlayer):
#         super(Angle_VQC_cir,self).__init__()

#         self.n_bit = n_bit
#         self.qlayer = qlayer
#         self.qm = QMachine(self.n_bit)
        
#         size = 2*( (self.n_bit -1) +(self.n_bit-2)//2 )
#         parm_num = self.qlayer*size
#         self.w = Parameter([parm_num,1]).reshape([self.qlayer,size//2,2]).toGPU()
        
#         self.paras = Parameter([
#             self.qlayer*self.n_bit*3,1
#         ]).reshape([self.qlayer,self.n_bit,3]).toGPU()

#         obs_list = [{
#             'wires':[i],
#             'observables':['Z'],
#             'coefficient':[1]
#         }for i in range(0,self.n_bit)]

#         self.ma = MeasureAll(obs=obs_list)

#     def encode_input(self, input, qm, encoding='ry'):
#         for i in range(self.n_bit):
#             if encoding == 'ry':
#                 RY(qm, wires=i, params=input[:,i].reshape([-1, 1]))
#             elif encoding == 'rx':
#                 RX(qm, wires=i, params=input[:,i].reshape([-1, 1]))
#             elif encoding == 'rz':
#                 RZ(qm, wires=i, params=input[:,i].reshape([-1, 1]))
#             elif encoding == 'hadamard_ry':
#                 Hadamard(qm,wires=i)
#                 RY(qm, wires=i, params=input[:,i].reshape([-1, 1]))
#             elif encoding == 'hadamard_rx':
#                 Hadamard(qm,wires=i)
#                 RX(qm, wires=i, params=input[:,i].reshape([-1, 1]))
#         return qm
#     def encode_input_am(self, input, qm):
#         VQC_AmplitudeEmbedding(input, q_machine=qm)
#         return qm
    
#     def U1_layer(self,qm,w_parm):
#         for i in range(self.n_bit-1):
#             RX(qm,wires=i,params=w_parm[i][0])
#             RZ(qm,wires=i+1,params=w_parm[i][1])
#         for i in range(0,self.n_bit,2):
#             CZ(qm,wires=[i+1,i])
        
#         return qm

#     def U2_layer(self,qm,w_parm):
#         for j in range(0,self.n_bit-2,2):
#             bit_idx = int(j/2)
#             # print('self.n_bit-1+bit_idx:',self.n_bit-1+bit_idx)
#             RX(qm,wires=j,params=w_parm[self.n_bit-1+bit_idx][0])
#             RY(qm,wires=j+2,params=w_parm[self.n_bit-1+bit_idx][1])
        
#         for j in range(0,self.n_bit,4):
#             CZ(qm,wires=[j+2,j])
#         return qm
    
#     def qnn(self,qm):
#         for layer in range(self.qlayer):
#             w_parm = self.w[layer]
#             qm = self.U1_layer(qm,w_parm)
#             qm = self.U2_layer(qm,w_parm)
#         return qm
    
#     def vqc_layer(self,qm):
#         for layer in range(self.qlayer):
#             w_para = self.paras[layer]
#             for j in range(0,self.n_bit):
#                 w_para_ = w_para[j]
#                 RX(qm,wires=j,params=w_para_[0])
#                 RY(qm,wires=j,params=w_para_[1])
#                 # rz(qm,wires=j,params=w_para_[2])
#             for j in range(self.n_bit-1):
#                 CNOT(qm,wires=[j,j+1])
#             CNOT(qm,wires=[self.n_bit-1,0])
#         return qm
    
#     def forward(self,x, encoding='ry'):
#         self.qm.reset_states(x.shape[0])
#         qm = self.encode_input(x, self.qm, encoding=encoding)
         
#         # qm = self.qnn(qm)
#         qm = self.vqc_layer(qm)
#         out = self.ma(qm)
#         return out



class Angle_VQC_cir(QModule):
    def __init__(self, n_bit, qlayer):
        super(Angle_VQC_cir,self).__init__()
        self.n_bit = n_bit
        self.qlayer = qlayer
        self.qm = QMachine(self.n_bit)
      
        self.paras = Parameter((self.qlayer,self.n_bit,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
 
        self.ma = MeasureAll(obs=[{"Z0": 1},{"Z1": 1},{"Z2": 1},{"Z3": 1},{"Z4": 1},{"Z5": 1}])

    def encode_input(self, input, qm, encoding='ry'):
        for i in range(self.n_bit):
            if encoding == 'ry':
                ry(qm,wires=i, params=input[:,i].reshape([-1, 1]))
            elif encoding == 'rx':
                rx(qm,wires=i, params=input[:,i].reshape([-1, 1]))
            elif encoding == 'rz':
                rz(qm,wires=i, params=input[:,i].reshape([-1, 1]))
            elif encoding == 'hadamard_ry':
                hadamard(qm,wires=i)
                ry(qm,wires=i, params=input[:,i].reshape([-1, 1]))
            elif encoding == 'hadamard_rx':
                hadamard(qm,wires=i)
                rx(qm,wires=i, params=input[:,i].reshape([-1, 1]))
        return qm
  
    def vqc_layer(self,qm):
        for layer in range(self.qlayer):
            w_para = self.paras[layer]
            for j in range(0,self.n_bit):
                w_para_ = w_para[j]
                rx(qm,wires=j,params=w_para_[0])
                ry(qm,wires=j,params=w_para_[1])
            
            for j in range(self.n_bit-1):
                cnot(qm,wires=[j,j+1])
            cnot(qm,wires=[self.n_bit-1,0])
        return qm
    
    def forward(self,x, encoding='ry'):
        self.qm.reset_states(x.shape[0])
        qm = self.encode_input(x, self.qm, encoding=encoding)
        qm = self.vqc_layer(qm)
        out = self.ma(qm)
        return out


