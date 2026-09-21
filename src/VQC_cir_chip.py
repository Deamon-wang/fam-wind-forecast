import torch
import pyqpanda3.core as pq
import pyvqnet
from pyvqnet.qnn.vqc.torch import TorchQcloud3QuantumLayer

pyvqnet.backends.set_backend("torch-native")
 
import pyqpanda3.core as pq
from pyvqnet.qnn.pq3 import ProbsMeasure

def VQC_cir_QPanda3(input, param):
    n_wires = len(input)
    n_qlayers = int(len(param)/(int(n_wires) * 4))
    
    # params_ry1_dct, params_crx1_dct, params_ry2_dct, params_crx2_dct = distribute_parameters_from_array(param, n_wires, n_qlayers)

    circuit = pq.QCircuit()
    for i in range(n_wires):
        circuit << pq.RY(i, input[i])
    for k in range(n_qlayers):
        for i in range(n_wires):
            circuit << pq.RY(i, param[int(i + k * n_wires)])
        for i in range(n_wires - 1, -1, -1):
            circuit << pq.CRX(i, (i + 1) % n_wires, param[int(i + k * n_wires)])
        for i in range(n_wires):
            circuit << pq.RY(i, param[int(i + k * n_wires)])
        for i in [n_wires - 1] + list(range(n_wires - 1)):
            circuit << pq.CRX(i, (i - 1) % n_wires, param[int(i + k * n_wires)])

    prog = pq.QProg()
    prog << circuit
    
    machine = pq.CPUQVM()
    machine.run(prog, 1000)
    prob_list_result = machine.result().get_prob_list()
    return prob_list_result



def VQC_angle_cir(input, param):
    n_wires = len(input)
    n_qlayers = int(len(param)/(int(n_wires) * 2))
    # params_ry1_dct, params_crx1_dct, params_ry2_dct, params_crx2_dct = distribute_parameters_from_array(param, n_wires, n_qlayers)
    m_prog = pq.QProg()
    circuit = pq.QCircuit()
    # 编码层
    for i in range(n_wires):
        circuit << pq.RY(i, input[i])
    
    # 变分层
    for k in range(n_qlayers):
        for i in range(n_wires):
            circuit << pq.RX(i, param[int(i + k * n_wires + 0)])
            circuit << pq.RY(i, param[int(i + k * n_wires + 1)])
        for i in range(n_wires - 1):
            circuit << pq.CNOT(i,i+1)
        circuit << pq.CNOT(n_wires,0)

    m_prog<<circuit
    return m_prog
 
from pyvqnet.nn import Parameter
from pyvqnet.qnn.vqc.torch import QMachine, rx,ry,rz,cnot,RX, RY,RZ, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval,MeasureAll,QModule
from pyvqnet.qnn.vqc.torch import U3

from pyvqnet.tensor import QTensor
import numpy as np
# 用于虚拟机训练使用

from pyvqnet.nn.torch import TorchModule

class Angle_VQC_chip(TorchModule):
    def __init__(self, n_bit, qlayer):
        super(Angle_VQC_chip,self).__init__()
        self.n_bit = n_bit
        self.qlayer = qlayer
        self.qm = QMachine(self.n_bit)
        
        # 使用 ModuleList 存储每一层每一个比特的 U3 门，确保参数独立
      
        self.paras = Parameter((self.qlayer,self.n_bit,2),initializer=pyvqnet.utils.initializer.quantum_uniform)
        # 动态生成测量算子
        obs = [{f"Z{i}": 1} for i in range(self.n_bit)]
        self.ma = MeasureAll(obs=obs)

    def encode_input(self, input, qm, encoding='ry'):
        for i in range(self.n_bit):
            if encoding == 'ry':
                ry(qm,wires=i, params=input[:,i].reshape([-1, 1]))
        return qm
  
    def vqc_layer(self, qm):
        # 用 self.paras 驱动 rx/ry，使其进入 autograd 计算图、被 optimizer 更新。
        # 前向，导致量子线路参数完全不更新。
        # print('参数:',self.paras[0][0])
        
        for layer_idx in range(self.qlayer):
            w = self.paras[layer_idx]              # (n_bit, 2)
            for j in range(self.n_bit):
                rx(qm, wires=j, params=w[j, 0:1])
                ry(qm, wires=j, params=w[j, 1:2])
            for j in range(self.n_bit - 1):
                cnot(qm, wires=[j, j + 1])
            if self.n_bit > 1:
                cnot(qm, wires=[self.n_bit - 1, 0])
        return qm
    
    # pyvqnet 的量子门(ry 等)在 cuda + torch-native 后端下，会把传入的 torch 张量转成
    # cpu 的 QTensor 并丢失 autograd 连接。因此让整个量子层固定在 CPU 运行：拒绝随
    # model.to(cuda) 移动，forward 内把输入拉到 CPU，输出由调用方转回 GPU。
    def _apply(self, fn):
        return self
    def to(self, *args, **kwargs):
        return self

    def forward(self,x, encoding='ry'):
        if isinstance(x, torch.Tensor):
            x = x.cpu()
        self.qm.reset_states(x.shape[0])

        qm = self.encode_input(x, self.qm, encoding=encoding)
        qm = self.vqc_layer(qm)
        out = self.ma(qm)

        return out



 
