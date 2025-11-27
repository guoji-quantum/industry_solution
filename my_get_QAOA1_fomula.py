'''
Created by Kirihara on 2024/3/20
该程序用于获取任意IPMSP实例的1-level QAOA解析表达式
'''

import numpy as np

# 算子系数
def A_ij(Q, g, i, j):
    return (Q[i][j] + Q[j][i])/4
def B_i(Q, g, i):
    return g[i]/2 + np.sum(Q[:, i])/4 + np.sum(Q[i, :])/4

# 自定义三角函数
def c(beta):
    return np.cos(2 * beta)
def s(beta):
    return np.sin(2 * beta)

def c_1(Q, g, i, j, gamma):
    a_ij = A_ij(Q, g, i, j)
    return np.cos(2 * gamma * a_ij)
def s_1(Q, g, i, j, gamma):
    a_ij = A_ij(Q, g, i, j)
    return np.sin(2 * gamma * a_ij)

def c_2(Q, g, i, gamma):
    b_i = B_i(Q, g, i)
    return np.cos(2 * gamma * b_i)
def s_2(Q, g, i, gamma):
    b_i = B_i(Q, g, i)
    return np.sin(2 * gamma * b_i)

# 构造节点相邻节点编号list
def d(Q, i, j):
    d = []
    for u in range(Q.shape[0]):
        if u != i and u != j :
            d.append(u)
    return d
def d_1(Q, i):
    d_1 = []
    for u in range(Q.shape[0]):
        if u != i:
            d_1.append(u)
    return d_1

def e(Q, i, j):
    e = []
    for u in range(Q.shape[0]):
        if u != i and u != j :
            e.append(u)
    return e
def e_1(Q, i):
    e_1 = []
    for u in range(Q.shape[0]):
        if u != i:
            e_1.append(u)
    return e_1

def f_1(Q, i, j):
    f_1 = []
    for f in range(Q.shape[0]):
         if f != i and f != j :
            f_1.append(f)
    return f_1



def E_1(Q, g, i, j, beta, gamma):
    e_1 = c(beta) * s(beta) * s_1(Q, g, i, j, gamma)
    e_list = e(Q, i, j)
    for v in e_list:
        e_1 = e_1 * c_1(Q, g, v, j, gamma)
    e_1 = e_1 * c_2(Q, g, j, gamma)

    e_2 = c(beta) * s(beta) * s_1(Q, g, i, j, gamma)
    d_list = d(Q, i, j)
    for u in d_list:
        e_2 = e_2 * c_1(Q, g, i, u, gamma)
    e_2 = e_2 * c_2(Q, g, i, gamma)

    e_3 = c_2(Q, g, i, gamma)*c_2(Q, g, j, gamma) + s_2(Q, g, i, gamma)*s_2(Q, g, j, gamma)

    for u in d_list:
        e_3 = e_3 * (c_1(Q, g, i, u, gamma)*c_1(Q, g, u, j, gamma) + s_1(Q, g, i, u, gamma) * s_1(Q, g, u, j, gamma))

    e_4 = c_2(Q, g, i, gamma) * c_2(Q, g, j, gamma) - s_2(Q, g, i, gamma)*s_2(Q, g, j, gamma)
    for u in d_list:
        e_4 = e_4 * (c_1(Q, g, i, u, gamma) * c_1(Q, g, u, j, gamma) - s_1(Q, g, i, u, gamma) * s_1(Q, g, u, j, gamma))

    e_5 = (s(beta)*s(beta) / 2) * (e_3 - e_4)

    return e_1 + e_2 + e_5

def E_2(Q, g, i, beta, gamma):
    e_2 = s(beta) * s_2(Q, g, i, gamma)
    d_list = d_1(Q, i)
    for u in d_list:
        e_2 = e_2 * c_1(Q, g, i, u, gamma)
    return e_2

def QAOA_F(Q, g):
    def F(theta):
        beta = theta[:1][0]
        gamma = theta[1:2][0]
        N = Q.shape[0]

        E1 = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                E1 += A_ij(Q, g, i, j) * E_1(Q, g, i, j, beta, gamma)

        E2 = 0
        for i in range(N):
            E2 += B_i(Q, g, i) * E_2(Q, g, i, beta, gamma)

        return E1 + E2
    return F

def QAOA_F_ray(Q, g):
    def F(beta, gamma):
        N = Q.shape[0]

        E1 = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                E1 += A_ij(Q, g, i, j) * E_1(Q, g, i, j, beta, gamma)

        E2 = 0
        for i in range(N):
            E2 += B_i(Q, g, i) * E_2(Q, g, i, beta, gamma)

        return E1 + E2
    return F
