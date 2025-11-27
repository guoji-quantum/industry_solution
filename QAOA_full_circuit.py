from qiskit import QuantumCircuit
from QAOA_init import QAOA_init
from QAOA_cost import QAOA_cost
from QAOA_mixer import QAOA_mixer

class QAOA_full_circuit(object):
    """
    QAOA_full_circuit 类：用于构建完整的 QAOA 量子电路（initialization → cost → mixer 循环）。

    功能概述：
        - 根据 QUBO 参数 (Q, g) 和 QAOA 参数 (beta, gamma)，拼装出完整的 QAOA 电路。
        - 整个电路结构为：
            初始化电路  →  [cost 层 + mixer 层] × p  →  测量（若为 qasm 模式）

        - 初始化模块由 QAOA_init 决定（如均匀叠加、one-hot 初始化、问题自定义初始化等）。
        - cost 层由 QAOA_cost 生成，对应 e^{-i γ H_C}，根据 QUBO 的二次项与一次项构造。
        - mixer 层由 QAOA_mixer 生成，对应 e^{-i β H_M}，可选择标准 X-mixer 或自定义 mixer。
        - 支持 p 层结构，每层 cost 与 mixer 使用对应的 gamma[i]、beta[i]。
        - 根据 simulator_type 决定是否在最后添加测量。

    作用：
        该类是构建 QAOA 电路的核心入口，供 QAOA_black_box 或单独调用时使用。
    """
    def __init__(self, Q, g, n, m, p, beta, gamma, init_type, cost_type, mixer_type, simulator_type):
        self.Q = Q  # 二次项
        self.g = g  # 线性项
        self.n = n
        self.m = m
        self.p = p # QAOA 层数
        self.beta = beta # mixer operator 参数列表
        self.gamma = gamma # cost operator 参数列表
        # self.delta = delta # 改进mixer 参数列表
        self.init_type = init_type #初始化模块类型
        self.cost_type = cost_type
        self.mixer_type = mixer_type
        self.simulator_type = simulator_type

    def get_qaoa_circuit(self):
        N = self.n * self.m
        qaoa_circuit = QuantumCircuit(N, N)
        # 1. 初始化模块
        init_obj = QAOA_init(self.n, self.m, self.init_type)
        init_circuit = init_obj.get_init_circuit()
        qaoa_circuit.append(init_circuit, range(N))
        # 应用QAOA layer = cost + mixer
        for i in range(self.p):
            # 2. cost operator 模块
            cost_obj = QAOA_cost(self.Q, self.g, self.n, self.m, self.gamma[i], self.cost_type)
            cost_citcuit = cost_obj.get_cost_circuit()
            qaoa_circuit.append(cost_citcuit, range(N))
            # 3. mixer operator 模块
            mixer_obj = QAOA_mixer(self.n, self.m, self.beta[i],self.mixer_type)
            mixer_circuit = mixer_obj.get_mixer_circuit()
            qaoa_circuit.append(mixer_circuit, range(N))
            qaoa_circuit.barrier(range(N))
        if self.simulator_type == 'qasm_simulator':
            qaoa_circuit.measure(range(N), range(N))
        else:
            pass

        return qaoa_circuit