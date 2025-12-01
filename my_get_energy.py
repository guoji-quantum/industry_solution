'''
该程序用于获取能量值
'''
import numpy as np

def JS_obj (state, Q, g, c):
    x_sol = np.array([int(bit) for bit in state], dtype=np.uint8)
    c_term = np.sum(Q)/4 + np.sum(g)/2 + c
    q_dig = 0
    for d in range(Q.shape[0]):
        q_dig += Q[d][d]/4
#     print(np.dot( L_M_list[0], x_sol))
    try:
        max(x_sol)
#         print(x_sol)
        # Evaluates the cost distance from a binary representation of a path
        fun = (
            lambda x: np.dot(np.around(x), np.dot(Q, np.around(x)))
                      + np.dot(g, np.around(x))
                      + c - q_dig - c_term
        )
        cost = fun(x_sol)
    except:
        cost = 0

    return cost
