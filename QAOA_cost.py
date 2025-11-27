from qiskit import QuantumCircuit
import numpy as np

class QAOA_cost(object):
    """
    QAOA_cost 类用于构建 QAOA 的成本算子（Cost Hamiltonian）的量子电路部分。

    功能说明：
        - 根据给定的 QUBO 参数 Q（二次项矩阵）、g（线性项向量），
          以及 QAOA 参数 gamma，构建对应的 e^{-i * gamma * H_C} 电路。
        - H_C 对应 QUBO 形式： x^T Q x + g^T x。
        - 包含两类量子门：
              1. 二体项（二次项） → 使用 RZZ 门实现
              2. 一体项（线性项 + QUBO 归一化后产生的 Z 项）→ 使用 RZ 门实现
        - cost_type == 'standard' 时，构建完整的标准 QAOA 成本算子。
    
    输出：
        get_cost_circuit() 返回作用全目标比特的 cost 层 QuantumCircuit。
    """
    def __init__(self, Q, g, n, m, gamma, cost_type):
        self.Q = Q # 二次项
        self.g = g # 线性项
        self.n = n # job number
        self.m = m # machine number
        self.gamma = gamma  #cost operator参数
        self.cost_type = cost_type

    def append_zz_term(self, qc, q1, q2):
        #或者使用rzz
        qc.rzz(2*(self.Q[q1][q2]/4 + self.Q[q2][q1]/4)*self.gamma, q1, q2)

    def append_z_term(self, qc, q):
        # 求系数
        h_i = self.g[q]/2 + np.sum(self.Q[:,q])/4 + np.sum(self.Q[q,:])/4
        qc.rz(-2 * h_i * self.gamma, q)

    def get_cost_circuit(self):
        N = self.n * self.m
        qc = QuantumCircuit(N)
        if self.cost_type == 'standard':
            for ii in range(N-1):
                for jj in range(ii+1, N):
                    self.append_zz_term(qc, ii, jj)
            for i in range(N):
                self.append_z_term(qc, i)
        qc.barrier()
        return qc