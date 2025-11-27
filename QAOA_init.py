from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate
from math import sqrt
import math


class QAOA_init(object):
    """
    QAOA_init 类：用于构建 QAOA 的初始化电路（initial state preparation）。

    功能概述：
    - 根据 init_type 生成不同类型的 QAOA 初始态。
    - 支持两类初始化方式：

        1. standard —— 标准 QAOA 初始化：
            将所有比特准备为 |+>^{⊗N}，即对每个量子比特施加 Hadamard 门。
            这是普通 QAOA 使用的均匀叠加态。

        2. dicke_state —— 构造 Dicke 态的初始化：
            用于约束“一行 m 比特中恰有一个 1”（如 Job–Machine one-hot 编码）。
            通过 SCS（Symmetric Controlled Rotations）构造 Dicke 态，
            保证每个 job 子寄存器处于适合约束的初始叠加形式。

    - append_SCS_term() 是构建 Dicke 态的关键步骤：
            构建对称受控旋转（controlled RY），逐步将 k 个 |1⟩ 分布到多比特系统中。
            本质是文献中常用的 “SCS (Symmetric Conditional Swap) / Dicke-state preparation” 方法。

    输出：
    get_init_circuit() 返回一段初始化电路，可直接 append 到完整 QAOA 电路中。
    """
    def __init__(self, n, m, init_type):
        self.n = n   #job number
        self.m = m   #machine number
        self.init_type = init_type

    def append_SCS_term(self, m, l):
        N = self.n * self.m
        qc = QuantumCircuit(N, name='SCS_' + str(m) + ',' + str(l))
        m = m - 1
        for i in range(l):
            if (i + 1) == 1:
                qc.cx(m - 1, m)
                theta = sqrt(1 / (m + 1))
                c3ry_gate = RYGate(2 * math.acos(theta)).control(1)
                qc.append(c3ry_gate, [m, m - 1])
                qc.cx(m - 1, m)
            else:
                qc.cx(m - (i + 1), m)
                theta = sqrt((i + 1) / (m + 1))
                c3ry_gate = RYGate(2 * math.acos(theta)).control(2)
                qc.append(c3ry_gate, [m, m - (i + 1) + 1, m - (i + 1)])
                qc.cx(m - (i + 1), m)
        return qc

    def get_init_circuit(self):
        if self.init_type == 'standard':
            N = self.n * self.m
            qc = QuantumCircuit(N)
            qc.h(range(N))
        elif self.init_type == 'dicke_state':
            N = self.n * self.m
            qc = QuantumCircuit(N, name='dicke_init')
            for j in range(self.n):
                qc.x(N - j - 1)
            for i in range(N - 1):
                #         print(n-i)
                if N - i > self.n:
                    qc.append(self.append_SCS_term(N - i, self.n), range(N))
                else:
                    qc.append(self.append_SCS_term(N - i, (N - i - 1)), range(N))
                qc.barrier()
        return qc