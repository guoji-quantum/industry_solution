import numpy as np
import time
from qiskit import transpile
from qiskit_aer import Aer
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

from my_get_QUBO_term import get_QUBO_term
from my_get_QAOA1_fomula import QAOA_F
from my_get_energy import JS_obj
from QAOA_full_circuit import QAOA_full_circuit


def solve_instance(n, m, inst_L, n_runs=20, optimizers=None):
    """
    通过多次随机初始化运行 QAOA，对给定规模 (n,m) 的问题实例进行求解。

    参数：
        n (int): 作业数量（job 数）
        m (int): 机器数量（machine 数）
        inst_L (list): 每个 job 的长度（加工时间）
        n_runs (int): 随机初始化次数，每次运行 QAOA 求解
        optimizers (list[str]): 要使用的经典优化器名称列表

    返回：
        results (dict): 每个优化器在多次初始化下的最优参数与能量
        Q, g, c : QUBO 的三项参数，供后续计算使用
    """
    if optimizers is None:
        optimizers = ['COBYLA', 'Nelder-Mead', 'Powell', 'BFGS', 'SLSQP']

    Q, g, c = get_QUBO_term(n, m, inst_L, 100, 10)
    obj_function = QAOA_F(Q, g)

    results = {f"N_{n}M_{m}": {}}
    for optimizer in optimizers:
        results[f"N_{n}M_{m}"][optimizer] = []

    for i_run in range(n_runs):
        print('run' + str(i_run))
        beta_min_limit = 0
        beta_max_limit = 0.5 * np.pi
        gamma_min_limit = 0
        gamma_max_limit = 2 * np.pi
        
        beta = np.random.uniform(beta_min_limit, beta_max_limit, (1, 1))
        gamma = np.random.uniform(gamma_min_limit, gamma_max_limit, (1, 1))
        init_point = np.hstack((beta, gamma))[0]
        print('随机初始点' + str(init_point))
        
        for optimizer in optimizers:
            print(f'optimizer={optimizer}')
            start = time.process_time()
            res_sample = minimize(obj_function, init_point, method=optimizer, options={'maxiter': 500, 'disp': False})
            end = time.process_time()
            best_energy = res_sample['fun']
            best_beta = res_sample['x'][0]
            best_gamma = res_sample['x'][1]

            results[f"N_{n}M_{m}"][optimizer].append({
                "run": i_run,
                "beta": best_beta,
                "gamma": best_gamma,
                "energy": best_energy
            })
    return results, Q, g, c

def invert_counts(counts):
    """
    将 qiskit 测量结果的 bitstring 从右到左反转（Qiskit 默认最低位在右）。

    参数：
        counts (dict): {'bitstring': count}

    返回：
        dict: {'反转后的bitstring': count}
    """
    return {k[::-1]: v for k, v in counts.items()}

def compute_JS_energy(counts, Q, g, c):
    """
    根据测量结果计算平均 JS (Job Scheduling) 能量。

    参数：
        counts (dict): 量子测量结果，bitstring -> count
        Q, g, c: QUBO 形式中的三类参数

    返回：
        float: 依据测量分布得到的平均能量
    """
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = JS_obj(meas, Q, g, c)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy / total_counts

def run_qaoa_with_params(n, m, inst_L, beta, gamma, shots=10000, 
                         init_type='standard', p=1, cost_type='standard', 
                         mixer_type='standard', simulator_type='qasm_simulator'):
    """
    在指定 (beta, gamma) 参数下构建并运行 QAOA 电路。

    步骤：
        1. 生成 QUBO 参数
        2. 构建 QAOA 电路
        3. 使用 qasm_simulator 采样
        4. 计算对应能量

    参数：
        n, m, inst_L: 问题规模与 job 长度
        beta, gamma: QAOA 参数
        shots (int): 量子测量次数
        init_type, cost_type, mixer_type: 电路构建方式
        p (int): QAOA 层数
        simulator_type (str): 后端类型

    返回：
        avg_energy (float): 平均能量
        counts (dict): 测量结果
        Q, g, c: QUBO 参数
    """
    Q, g, c = get_QUBO_term(n, m, inst_L, 100, 10)
    qaoa_obj = QAOA_full_circuit(Q, g, n, m, p, beta, gamma, init_type, cost_type,
                                 mixer_type, simulator_type)
    qaoa_circuit = qaoa_obj.get_qaoa_circuit()

    backend = Aer.get_backend('qasm_simulator')
    transpile_qc = transpile(qaoa_circuit, backend=backend)
    counts = backend.run(transpile_qc, shots=shots).result().get_counts()
    avg_energy = compute_JS_energy(invert_counts(counts), Q, g, c)
    return avg_energy, counts, Q, g, c

def is_feasible(bitstring, n, m):
    """
    判断 bitstring 是否为可行调度解：
    每个 job 对应的 m 个 bit 中必须恰好有一个 '1'。

    参数：
        bitstring (str): 长度 n*m 的二进制串
        n, m (int): job 数与 machine 数

    返回：
        bool: True 表示可行；False 表示不可行
    """
    if len(bitstring) != n*m:
        return False
    for j in range(n):
        segment = bitstring[j*m:(j+1)*m]
        if segment.count('1') != 1:
            return False
    return True

def get_job_assignment(bitstring, n, m):
    """
    在确认可行解后，返回每个 job 分配到的机器编号。

    参数：
        bitstring (str): 二进制表示的调度方案
        n, m (int): job 数与 machine 数

    返回：
        list[int]: job_assignment[j] = job j 分配的机器编号
    """
    job_assignment = []
    for j in range(n):
        segment = bitstring[j*m:(j+1)*m]
        assigned_machine = segment.index('1')
        job_assignment.append(assigned_machine)
    return job_assignment

def assign_jobs(n, m, inst_L):
    """
    使用启发式策略给 job 分配机器，并生成对应 bitstring。

    策略：
        1. 计算作业平均时长，区分大作业与普通作业
        2. 若超过 1 个“大作业”，仅保留 job_id 最小的一个，其余并入普通作业
        3. 将唯一的大作业分配到机器0
        4. 其他作业按长度降序排序，逐个分配给当前负载最小的机器
        5. 最终构建 bitstring 和对应十进制值

    参数：
        n (int): job 数
        m (int): machine 数
        inst_L (list): job 时长

    返回：
        machine_assignments (list[list]): 每台机器的作业列表
        bitstring (str): 对应的二进制调度方案
        decimal_value (int): bitstring 的十进制表示
    """
    # 计算平均作业长度
    total_length = sum(inst_L)
    avg_job_duration = total_length / n

    # 每个作业的数据结构： (job_id, length)
    jobs = [(i, inst_L[i]) for i in range(n)]

    # 找出大于平均值的作业
    big_jobs = [job for job in jobs if job[1] > avg_job_duration]
    other_jobs = [job for job in jobs if job[1] <= avg_job_duration]

    # 对大作业排序（可选步骤，让分配确定性更好）
    big_jobs.sort(key=lambda x: x[1], reverse=True)

    # 分配结果：machine_assignments[machine] = list of (job_id, length)
    machine_assignments = [[] for _ in range(m)]

    # 如果有big_jobs，先分配big_jobs
    # 如果big_jobs数量不为1时处理逻辑
    if len(big_jobs) > 1:
        # 从big_jobs中选取编号最小的一个作为最终的big_job
        chosen_big_job = min(big_jobs, key=lambda x: x[0])
        # 剩下的big_jobs并入other_jobs
        remaining_big_jobs = [jb for jb in big_jobs if jb != chosen_big_job]
        other_jobs.extend(remaining_big_jobs)
        big_jobs = [chosen_big_job]
    # 如果big_jobs为0或1则不需要特别处理
    # big_jobs == 0 时正常跳过分配big_job步骤
    # big_jobs == 1 时正常执行原逻辑

    # 分配结果：machine_assignments[machine] = list of (job_id, length)
    machine_assignments = [[] for _ in range(m)]

    # 如果有big_jobs，先分配big_job
    if big_jobs:
        # 假设big_jobs数量此时必为1
        # 将该big_job分配给第0台机器（或其他策略）
        machine_assignments[0].append(big_jobs[0])

    # 分配其他作业（包括当没有big_jobs时所有的jobs）
    # 将other_jobs中的作业按长度从大到小排序
    other_jobs.sort(key=lambda x: x[1], reverse=True)

    for job in other_jobs:
        # 找出当前负载最小的机器（若有并列选编号最小的机器）
        machine_loads = [sum(j[1] for j in machine_assignments[mm]) for mm in range(m)]
        min_load = min(machine_loads)
        target_machine = machine_loads.index(min_load)

        # 将该作业分配给这台机器
        machine_assignments[target_machine].append(job)

    # 分配完成后，构建最终bitstring和decimal_value
    job_to_machine = [-1]*n
    for mm in range(m):
        for (j_id, length) in machine_assignments[mm]:
            job_to_machine[j_id] = mm

    # 构建二进制串
    bitstring_list = ['0']*(n*m)
    for j_id, assigned_machine in enumerate(job_to_machine):
        bit_index = j_id*m + assigned_machine
        bitstring_list[bit_index] = '1'
    bitstring = "".join(bitstring_list)

    # 转十进制
    decimal_value = int(bitstring, 2)

    return machine_assignments, bitstring, decimal_value
def plot_gantt_chart(bitstring, n, m, inst_L):
    """
    根据调度 bitstring 绘制甘特图。

    步骤：
        1. 从 bitstring 得到 job → machine 的映射
        2. 计算每台机器的作业时间轴
        3. 构建 DataFrame
        4. 使用 plotly 生成交互式甘特图

    参数：
        bitstring (str): job 调度方案
        n, m (int): job 数与机器数
        inst_L (list[int]): job 时长列表

    返回：
        None（直接展示图形）
    """    
    job_assignment = get_job_assignment(bitstring, n, m)
    current_time = [0]*m

    records = []
    # 将基准日期设为当天(零点)
    today = datetime.today()
    start_date = datetime(today.year, today.month, today.day)

    for j in range(n):
        assigned_machine = job_assignment[j]
        duration = inst_L[j]
        start = current_time[assigned_machine]
        end = start + duration
        current_time[assigned_machine] = end

        start_dt = start_date + timedelta(hours=start)
        end_dt = start_date + timedelta(hours=end)

        # 这里的Task就是作业号，将其用于color字段，这样每个Task都会有独特的颜色
        records.append(dict(
            Task=f"工件{j}",
            Start=start_dt,
            Finish=end_dt,
            Resource=f"机器{assigned_machine}"
        ))

    df = pd.DataFrame(records)

    # 使用Task作为color字段，自动分配不同颜色
    fig = px.timeline(df, 
                      x_start="Start", 
                      x_end="Finish", 
                      y="Resource", 
                      color="Task",
                      text="Task",
                      color_discrete_sequence=px.colors.qualitative.T10  # 可换为你喜欢的配色方案
                     )

    # 倒序y轴，使上方是第一台机器
    fig.update_yaxes(autorange="reversed")

    # 更新布局和字体
    fig.update_layout(
        title="工件分配甘特图",
        xaxis_title="运行时间",
        yaxis_title="机器",
        font=dict(
            family="SimHei",  # 替换为本机可用的中文字体
            size=12
        ),
        legend_title_text="作业编号"  # 图例标题
    )

    # 将x轴刻度转化成相对于当日0点的小时数（可选步骤）
    max_end_time = max(current_time)
    tickvals = [start_date + timedelta(hours=h) for h in range(max_end_time+1)]
    ticktext = [str(h) for h in range(max_end_time+1)]
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

    fig.show()

def plot_bar_chart(keys, values, colors):
    """
    绘制前若干个解的柱状图（可区分可行/不可行）。

    参数：
        keys (list): x轴的整数标签（通常为解的十进制值）
        values (list): 各解的计数
        colors (list): 每根柱子的颜色，用于突出可行解

    返回：
        None（直接展示图形）
    """
    # 构建DataFrame
    df = pd.DataFrame({"keys": keys, "values": values})

    # 使用px.bar绘制柱状图，这里暂时不使用color映射分类，而是直接在后续使用update_traces指定colors
    fig = px.bar(df, x="keys", y="values")

    # 更新颜色和边框
    fig.update_traces(
        marker_color=colors,
        marker_line_color='black',
        marker_line_width=1
    )

    # 更新布局，包括中文字体、标题、轴标签等
    fig.update_layout(
        title="前30个解分布（红色为可行解）",
        xaxis_title="十进制表示",
        yaxis_title="计数",
        font=dict(
            family="SimHei",  # 请替换为本机可用中文字体
            size=12
        )
    )

    # X轴刻度旋转
    fig.update_xaxes(tickangle=45)

    # 添加Y轴网格线
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.3)')

    fig.show()

