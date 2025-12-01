# 算法流程

1. 构建 IPMSP 的 QUBO 数学模型

调用 get_QUBO_term() 生成二次项矩阵 Q、线性项 g 以及常数项 c，将调度约束编码为惩罚项。

2. 构建单层或多层 QAOA 量子线路

使用 QAOA_full_circuit 依次拼装：

  - 初始化模块 QAOA_init（|+⟩ 态或 Dicke 初始化） 

  - 成本算子 QAOA_cost（由 QUBO Hamiltonian 构建） 

  - Mixer 算子 QAOA_mixer（标准/XY-ring/并行 XY/full XY 多种类型） 

3. 设置 QAOA 初始参数

在 solve_instance 中为每次运行随机初始化 β、γ（或更深层时为多维向量）。

4. 运行包含参数的 QAOA 电路

在 run_qaoa_with_params() 中执行：

  - 构建 QAOA（由 QAOA_full_circuit 完成）

  - transpile 量子线路

  - 在 qasm_simulator 上采样

并依据 counts 结果计算平均调度能量。

5. 测量量子比特并得到 IPMSP 解

  - 将 Qiskit 默认的 bitstring 反转（低位在右）

  - 对所有测量样本计算调度能量：compute_JS_energy()

  - 判断可行性 (is_feasible) 与机器分配 (get_job_assignment)

6. 判断停止条件

若满足能量收敛或达到最大迭代次数 → 输出当前最优解。

否则使用 Powell 或指定优化器继续更新 β、γ。

7. 输出最优调度方案

包括：

  - 最优 β、γ

  - 最低能量值

  - 对应调度 bitstring 的可行性与机器分配结果

  - 可视化甘特图与可行解柱状图（heuristic + QAOA
