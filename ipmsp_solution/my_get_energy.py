import numpy as np
def JS_obj (state, Q, g, c):
    """
    JS_obj: 给定一个二进制状态 state 以及 QUBO 参数 (Q, g, c)，
    计算对应的调度目标值（能量），并做了一次常数平移（减去 q_dig 和 c_term），
    通常用于把“最优解”的目标值归一到 0 附近，便于比较和优化。
    """
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