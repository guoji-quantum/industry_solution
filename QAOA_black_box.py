from qiskit import transpile
from qiskit_aer import Aer
import qiskit
import matplotlib.pyplot as plt
import numpy as np
from QAOA_full_circuit import QAOA_full_circuit

class QAOA_black_box(object):
    """
    QAOA_black_box 类：用于将 QAOA 整个求解过程封装成“可被优化器直接调用的黑盒函数”。

    主要功能：
    1. 以 QUBO (Q, g, c) 为代价函数，构建 QAOA 电路并返回期望能量。
    2. 对输入角度 θ（包含所有 beta 和 gamma）运行 QAOA：
            θ = [beta_1 ... beta_p, gamma_1 ... gamma_p]
    3. 自动构建 QAOA 电路（调用 QAOA_full_circuit）。
    4. 在 qasm_simulator 上运行，得到测量分布 counts。
    5. 根据 counts 计算 JS 能量（作为优化目标）。
    6. 返回一个“黑盒”函数 f(theta)，便于外部优化算法直接调用。

    换句话说：
    —— QAOA_black_box 就是把 QAOA 的参数求解问题变成：最小化 f(theta)。
    —— 外部优化器（GA、CMA-ES、PSO、Bayesian Optimization 等）只需要不断询问 f(theta) 的值即可。

    注意：
    - 支持 qasm_simulator（测量概率形式）。
    - 若未来需要 Statevector，可扩展 compute_VRP_energy_sv。
    - 包含 bitstring ↔ 数值、反序、statevector 转概率等辅助函数。
    """
    def __init__(self, Q, g, c, n,m, p, init_type, cost_type, mixer_type, simulator_type):
        self.Q = Q  # 二次项
        self.g = g  # 线性项
        self.c = c  # 常数项
        self.n = n  #
        self.m = m
        self.p = p  # QAOA 层数
        self.init_type = init_type  # 初始化模块类型
        self.cost_type = cost_type
        self.mixer_type = mixer_type
        self.simulator_type = simulator_type
    def JS_obj (self, state):
        x_sol = np.array([int(bit) for bit in state], dtype=np.uint8)
        try:
            max(x_sol)
            # Evaluates the cost distance from a binary representation of a path
            fun = (
                lambda x: np.dot(np.around(x), np.dot(self.Q, np.around(x)))
                          + np.dot(self.g, np.around(x))
                          + self.c
            )
            cost = fun(x_sol)
        except:
            cost = 0
        return cost
    def invert_counts(self, counts):
        return {k[::-1]: v for k, v in counts.items()}

    def compute_JS_energy(self, counts):
        energy = 0
        total_counts = 0
        for meas, meas_count in counts.items():
            obj_for_meas = self.JS_obj(meas)
            energy += obj_for_meas * meas_count
            total_counts += meas_count
        return energy / total_counts

    def state_num2str(self,basis_state_as_num, nqubits):
        return '{0:b}'.format(basis_state_as_num).zfill(nqubits)


    def state_str2num(self, basis_state_as_str):
        return int(basis_state_as_str, 2)

    def state_reverse(self, basis_state_as_num, nqubits):
        basis_state_as_str = self.state_num2str(basis_state_as_num, nqubits)
        new_str = basis_state_as_str[::-1]
        return self.state_str2num(new_str)

    def get_adjusted_state(self, state):
        nqubits = np.log2(state.shape[0])
        if nqubits % 1:
            raise ValueError("Input vector is not a valid statevector for qubits.")
        nqubits = int(nqubits)

        adjusted_state = np.zeros(2 ** nqubits, dtype=complex)
        for basis_state in range(2 ** nqubits):
            adjusted_state[self.state_reverse(basis_state, nqubits)] = state[basis_state]
        return adjusted_state

    def state_to_ampl_counts(self, vec, eps=1e-15):
        """Converts a statevector to a dictionary
        of bitstrings and corresponding amplitudes
        """
        qubit_dims = np.log2(vec.shape[0])
        if qubit_dims % 1:
            raise ValueError("Input vector is not a valid statevector for qubits.")
        qubit_dims = int(qubit_dims)
        counts = {}
        str_format = '0{}b'.format(qubit_dims)
        for kk in range(vec.shape[0]):
            val = vec[kk]
            if val.real ** 2 + val.imag ** 2 > eps:
                counts[format(kk, str_format)] = val
        return counts

    def compute_VRP_energy_sv(self,sv):
        """Compute objective from statevector
        For large number of qubits, this is slow.
        """
        counts = self.state_to_ampl_counts(sv)
        counts_invert = self.invert_counts(counts)
        return sum(self.VRP_obj(np.array([int(x) for x in k])) * (np.abs(v) ** 2) for k, v in counts_invert.items())

    def get_black_box(self):
        def f(theta):
            # theta 是待优化参数列表，不用传参
            beta = theta[:self.p]
            gamma = theta[self.p:2 * self.p]
            # delta = theta[2 * self.p:]
            qaoa_obj = QAOA_full_circuit(self.Q, self.g, self.n , self.m, self.p, beta, gamma, self.init_type, self.cost_type,
                                         self.mixer_type, self.simulator_type)
            qaoa_circuit = qaoa_obj.get_qaoa_circuit()
            # qaoa_circuit.draw()
            plt.show()
            if self.simulator_type == 'qasm_simulator':

                backend = Aer.get_backend('qasm_simulator')
                transpiled_qc = transpile(qaoa_circuit, backend)
                result = backend.run(transpiled_qc).result()
                counts = result.get_counts()
                return self.compute_JS_energy(self.invert_counts(counts))
            else:
                # backend = Aer.get_backend('statevector_simulator')
                # sv = execute(qaoa_circuit, backend).result().get_statevector()
                # return self.compute_VRP_energy_sv(sv)
                pass
        return f
