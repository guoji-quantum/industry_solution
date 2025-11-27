'''
Created by Kirihara on 2024/3/20
该程序用于 计算二次项系数、线性系数、常数项
'''

import numpy as np


def get_QUBO_term(n, m, inst_L, a, b):
    # 定义两个列向量
    # LM(m)：第i个machine上全部job的运行时间
    L_M_list = [[] for i in range(m)]
    for i in range(m):
        L_M = [0 for ii in range(n * m)]
        for j in range(n):
            L_M[j * m + i] = inst_L[j]
        L_M_list[i] = L_M
    # Z_J(i): 第i个job在全部machine上的运行时间
    Z_J_list = [[] for i in range(n)]
    for i in range(n):
        Z_J = [0 for ii in range(n * m)]
        for j in range(m):
            Z_J[i * m + j] = 1
        Z_J_list[i] = Z_J
    # 计算惩罚因子
    A = np.max(inst_L) * a
    B = np.max(inst_L) * b
    # A = 10
    # B = 10
    # A = 1
    # B = 1
    # 计算二次项
    term1 = 0
    term2 = 0
    Q = 0
    for i in range(n):
        # print(f'i={i},Z_J_list[i]={Z_J_list[i]}')
        term1 += A * np.outer(Z_J_list[i], Z_J_list[i])
    for j in range(m):
        # print(f'm={j},L_M_list[j]={L_M_list[j]}')
        term2 += B * np.outer(L_M_list[j], L_M_list[j]) + B * np.outer(L_M_list[0], L_M_list[0]) - 2 * B * np.outer(
            L_M_list[j], L_M_list[0])
    Q = term1 + term2

    term3 = [0 for i in range(n * m)]
    term3 = np.array(term3)
    g = 0
    for i in range(n):
        Z_J_list[i] = np.array(Z_J_list[i])
        term3 += 2 * A * Z_J_list[i]
    g = L_M_list[0] - term3

    # 计算附加项
    c = A * n

    return  Q, g, c