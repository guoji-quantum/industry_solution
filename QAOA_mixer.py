import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

class QAOA_mixer(object):
    """
    QAOA_mixer 类：用于构建 QAOA 的 mixer operator（混合算子）电路。

    功能概述：
        - 根据 mixer_type 构造不同风格的 mixer（H_M），其作用是探索解空间。
        - 在 QAOA 中，mixer 决定了从初始态可以到达哪些状态，对搜索结构影响巨大。

    支持的 mixer 类型：

    1. 'standard'
        标准 QAOA mixer：对每个 qubit 做 RX(2β)
        对应 H_M = Σ X_i，是传统 QAOA 的默认选择。

    2. 'XY_ring'
        环形 XY-mixer，只耦合最近邻 (i, i+1)：
        对每对比特施加 e^{-i β (X_i X_j + Y_i Y_j)}。
        常用于一维环结构、粒子数保持等情况。

    3. 'XY_par_ring'
        并行 ring mixer：偶数与奇数索引分别并行耦合，
        作用是在深度受限硬件中实现更高的并行性。

    4. 'XY_full'
        全耦合 XY-mixer：构造所有可能的成对 XY 旋转，
        允许从任意 basis 的多个 Hamming weight subspace 进行跳转，
        适用于需要强探索能力的组合优化问题。

    备注：
        - append_XY_mixer_term 实现 e^{-i β (X_i X_j + Y_i Y_j)} 的标准分解（利用 RXX+RYY 的组合转化）。
        - get_mixer_circuit() 返回一段 QuantumCircuit，可直接接入 QAOA 层结构。
    """
    def __init__(self,n, m, beta, mixer_type):
        self.n = n*m
        # self.m = m
        self.beta = beta

        self.mixer_type = mixer_type

    def append_XY_mixer_term(self, qc, q1, q2):
        qc.rx(-np.pi / 2, q1)
        qc.rx(np.pi / 2, q2)
        qc.cx(q1, q2)
        qc.rx(-2 * self.beta, q1)
        qc.rx(2 * self.beta, q2)
        qc.cx(q1, q2)
        qc.rx(np.pi / 2, q1)
        qc.rx(-np.pi / 2, q2)
        qc.barrier()


    def get_mixer_circuit(self):
        if self.mixer_type == 'XY_ring':
            # 1 MXY_ring mixer
            pair_list = [(i % self.n, (i + 1) % self.n) for i in range(0, self.n)]
            # print('XY_ring' + str(pair_list))
            qc = QuantumCircuit(self.n, name=self.mixer_type)
            for i in range(len(pair_list)):
                self.append_XY_mixer_term(qc, pair_list[i][0], pair_list[i][1])
        elif self.mixer_type == 'XY_par_ring':
        # 2 MXY_Par_ring mixer
            n0 = max(self.n + (self.n % 2) - 1, 2)  # 最大奇数不大于n的计算
            n1 = max(self.n - (self.n % 2), 1)  # 最大偶数不大于n的计算
            pair_list = [(i, (i + 1) % self.n) for i in range(0, n0, 2)] + [(i, (i + 1) % self.n) for i
                                                                                     in range(1, n1, 2)]
            # print('XY_par_ring' + str(pair_list))
            qc = QuantumCircuit(self.n, name=self.mixer_type)
            for i in range(len(pair_list)):
                self.append_XY_mixer_term(qc, pair_list[i][0], pair_list[i][1])
        elif self.mixer_type == 'XY_full':
            # 3 MXY_full
            m = self.n + self.n % 2 - 1
            k = int((m + m % 2 - 2) / 2)
            full_pair_list = [[] for i in range(m)]
            # 使用嵌套循环遍历数字的所有组合
            for i in range(0, m):
                for j in range(i + 1, m):
                    full_pair_list[(i + j) % m].append((i, j))
            # 创建一个集合，用于记录已经出现的数字
            pair_list = []
            for c in range(m):
                existing_numbers = set()
                # 遍历数对列表，将已经出现的数字添加到集合中
                for pair in full_pair_list[c]:
                    existing_numbers.update(pair)
                # 遍历数字范围，并判断缺失的数字对
                for i in range(0, self.n):
                    for j in range(i + 1, self.n):
                        pair_add = (i, j)
                        if i not in existing_numbers and j not in existing_numbers:
                            full_pair_list[c].append(pair_add)
                pair_list.extend(full_pair_list[c])
            qc = QuantumCircuit(self.n, name=self.mixer_type)
            for i in range(len(pair_list)):
                self.append_XY_mixer_term(qc, pair_list[i][0], pair_list[i][1])
        elif self.mixer_type == 'standard':
            qc = QuantumCircuit(self.n, name='standard_mixer')
            for i in range(self.n):
                qc.rx(2 * self.beta, i)
            qc.barrier()
        return qc
