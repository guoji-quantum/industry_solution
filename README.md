# 车间调度问题算法流程

1. 开始

2. 构建 IPMSP 数学模型
   
    - 调用 get_QUBO_term() 生成二次项矩阵 Q、线性项 g 以及常数项 c，将调度约束编码为惩罚项。

3. 构建单层 QAOA 量子线路，使用 QAOA_full_circuit 依次拼装

    - 初始化模块 QAOA_init（|+⟩ 态或 Dicke 初始化）； 
        
    - 成本算子 QAOA_cost（由 QUBO Hamiltonian 构建）；
        
    - Mixer 算子 QAOA_mixer（标准/XY-ring/并行 XY/full XY 多种类型）；

4. 设置 QAOA 初始参数

    - 在 solve_instance 中为每次运行随机初始化 β、γ（或更深层时为多维向量）

5. 运行含优化参数的 QAOA 量子线路
    - 在 run_qaoa_with_params() 中执行：

        - 构建 QAOA（由 QAOA_full_circuit 完成）

        - transpile 量子线路

        - 在 qasm_simulator 上采样

    - 依据 counts 结果计算平均调度能量。

6. 测量量子比特，获得 IPMSP 解

    - 将 Qiskit 默认的 bitstring 反转（低位在右）

    - 对所有测量样本计算调度能量：compute_JS_energy()
      
    - 判断可行性 (is_feasible) 与机器分配 (get_job_assignment)

7. 判断是否达到停止条件

    - 若满足能量收敛或达到最大迭代次数 → 输出最优解 → 结束

    - 否 → 使用 Powell 优化器优化参数 → 返回步骤 4

8. 输出最优调度方案

    - 最优 β、γ

    - 最低能量值

    - 对应调度 bitstring 的可行性与机器分配结果

    - 可视化甘特图与可行解柱状图（heuristic + QAOA）
