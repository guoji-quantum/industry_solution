import numpy as np
import matplotlib.pyplot as plt
from pyqubo import Model
import networkx as nx


def calculate_route_cost(route, distance_matrix):
    """
    计算路线的总成本（距离）。
    """
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance


def generate_customer_matrix(n_customers, n_routes, routes):
    """
    生成二进制矩阵，指示哪些客户由哪些路线服务。
    """
    customer_matrix = np.zeros((n_customers, n_routes))
    for k, route in enumerate(routes):
        for i in range(1, n_customers + 1):
            if i in route:
                customer_matrix[i - 1, k] = 1
    return customer_matrix


def qubo_to_matrix(qubo):
    """
    将QUBO字典转换为QUBO矩阵。
    """
    n = len(set(i for i, j in qubo.keys()))
    qubo_matrix = np.zeros((n, n))
    for k, v in qubo.items():
        c_index = int(k[0][2:-1])
        r_index = int(k[1][2:-1])
        qubo_matrix[c_index, r_index] = v
    return qubo_matrix


def get_ising_from_model(model, feed_dict):
    """
    从PyQUBO模型获取伊辛模型参数。
    """
    h, J, offset = model.to_ising(feed_dict=feed_dict)
    return h, J, offset


def ising_to_matrix(h, J):
    """ising 模型转化为 ising 矩阵"""
    n = len(h)
    ising_matrix = np.zeros((n, n))
    for k, v in h.items():
        index = int(k[2:-1])
        ising_matrix[index, index] = v
    for k, v in J.items():
        i, j = k.split(",")
        i_index = int(i[2:-1])
        j_index = int(j[2:-1])
        ising_matrix[i_index, j_index] = v / 2
        ising_matrix[j_index, i_index] = v / 2
    return ising_matrix


def check_route_validation(result, demands, n_nodes, n_vehicles, capacity):
    """
    验证求解的结果路径是否满足所有约束。
    """
    is_feasible = []
    content_details = []
    chosen_routes_details = []
    for k, v in result.items():
        chosen_routes_details.append(list(eval(k)))

    # 1. 检查选择的路径数量是否等于小车数量
    if len(chosen_routes_details) != n_vehicles:
        is_valid1 = False
        content1 = f"小车选择路径数量 ({len(chosen_routes_details)}) 与车辆数量 ({n_vehicles}) 不符。"
        print(content1)
    else:
        is_valid1 = True
        content1 = f"小车选择路径数量 ({len(chosen_routes_details)}) 与车辆数量 ({n_vehicles}) 相符。"
    is_feasible.append(is_valid1)
    content_details.append(content1)

    customer_nodes = set(range(1, n_nodes))  # 假设节点0是仓库
    visited_customers = {}

    is_valid2 = True
    is_valid3 = True
    route_invalid = []
    route_invalid_d = []
    for route in chosen_routes_details:
        if not route or route[0] != 0 or route[-1] != 0:
            is_valid2 = False
            route_invalid_d.append(route)

        current_route_capacity = 0
        for i in range(len(route) - 1):
            node = route[i]
            if node != 0:  # 非仓库节点
                visited_customers[node] = visited_customers.get(node, 0) + 1
                current_route_capacity += demands.get(
                    node, 0
                )  # 累加需求，如果节点不在demands中，则需求为0

        if current_route_capacity > capacity:
            is_valid3 = False
            route_invalid.append(route)

    # 2. 检查所有路径都以仓库节点开头和结尾
    if is_valid2:
        content2 = f"所有路径均以仓库节点开头和结尾。"
    else:
        content2 = f"路径{route_invalid_d}未以仓库节点开头或结尾。"
        print(content2)
    is_feasible.append(is_valid2)
    content_details.append(content2)

    # 3. 检查所有选择的路径均不超过小车的容量限制
    if is_valid3:
        content3 = f"所有路径的总需求均不超过车辆容量。"
    else:
        content3 = f"路径{route_invalid}的总需求超过了车辆容量。"
        print(content3)
    is_feasible.append(is_valid3)
    content_details.append(content3)

    # 4. 检查每个客户节点是否被访问了且仅被访问一次
    is_valid4 = True
    node_lsit = []
    for customer_node in customer_nodes:
        if visited_customers.get(customer_node, 0) != 1:
            is_valid4 = False
            node_lsit.append(customer_node)
    if is_valid4:
        content4 = f"所有客户节点均被访问且仅被访问一次。"
    else:
        content4 = f"以下客户节点未被访问或被访问超过一次: {node_lsit}"
        print(content4)
    is_feasible.append(is_valid4)
    content_details.append(content4)

    # 5. 检查是否全覆盖客户节点
    if set(visited_customers.keys()) != customer_nodes:
        unvisited_customers = customer_nodes - set(visited_customers.keys())
        is_valid5 = False
        content5 = f"以下客户节点未被访问: {unvisited_customers}"
        print(content5)
    else:
        is_valid5 = True
        content5 = f"所有客户节点均被访问"
    is_feasible.append(is_valid5)
    content_details.append(content5)

    return is_feasible, content_details


def verify_solution(bitstring, candidate_routes, customers, n_vehicles):
    """
    验证求解结果是否违反优化问题中的两个约束。
    """
    chosen_routes = [candidate_routes[i] for i, bit in enumerate(bitstring) if bit == 1]

    # 验证车辆数量限制
    if len(chosen_routes) != n_vehicles:
        return False

    # 验证顾客访问
    customer_visits = {i: 0 for i in range(1, len(customers) + 1)}
    for route in chosen_routes:
        for node in route:
            if node != 0:
                if node in customer_visits:
                    customer_visits[node] += 1
                else:
                    return False

    # 验证每个顾客点是否仅被访问一次
    for customer, count in customer_visits.items():
        if count != 1:
            return False

    return True


def print_solution(bitstring, candidate_routes, distance_matrix):
    """
    打印解决方案并返回选定的路线和总成本。
    """
    selected_routes = [candidate_routes[k] for k, v in enumerate(bitstring) if v == 1]
    total_cost = 0
    result = {}

    print("求解结果对应的路线:")
    for route in selected_routes:
        cost = calculate_route_cost(route, distance_matrix)
        result[str(tuple(route))] = cost
        total_cost += cost
        print(f"选择路线: {route} (路线长度: {cost})")

    print(f"路线总长度: {total_cost}")

    return result, total_cost


def plot_nodes(demands, coordinates):
    """
    绘制 CVRP 问题的客户节点和仓库节点图，节点大小根据demands值进行缩放。
    """
    plt.figure(figsize=(10, 8))

    coords = np.array(coordinates)

    # 仓库坐标（保持特殊标记）
    plt.scatter(coords[0][0], coords[0][1], c="red", s=200, marker="*", label="仓库")

    customer_demands = [demands.get(i, 0) for i in range(1, len(coords))]

    # 归一化demands值到合适的点大小范围（最小50，最大300）
    if len(customer_demands) > 0 and max(customer_demands) > 0:
        min_size = 50
        max_size = 300
        if max(customer_demands) == min(customer_demands):
            normalized_sizes = [100] * len(customer_demands)
        else:
            normalized_sizes = [
                min_size
                + (max_size - min_size)
                * (d - min(customer_demands))
                / (max(customer_demands) - min(customer_demands))
                for d in customer_demands
            ]
    else:
        normalized_sizes = [50] * len(coords[1:])

    # 客户节点（大小根据需求量调整）
    plt.scatter(coords[1:, 0], coords[1:, 1], c="blue", s=normalized_sizes)

    for i in range(1, len(coords)):
        plt.text(
            coords[i][0] + 0.5,
            coords[i][1] + 0.5,
            f"{i}({demands.get(i, 0)})",
            fontsize=10,
        )

    plt.title("CVRP nodes visualization")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    plt.legend()
    plt.grid()
    plt.show()


def plot_solution(
    solution,
    candidate_routes,
    coordinates=None,
    demands=None,
    data_source="generate dataset",
    output_file="../results/sa_solution.png",
    # distance_matrix=None,
):
    """
    绘制 CVRP 问题的可视化结果
    """
    r_indices = [int(k[2:-1]) for k, v in solution.items() if v == 1]
    selected_routes = [candidate_routes[i] for i in r_indices]

    plt.figure(figsize=(10, 8))

    # 定义路线的颜色
    colors = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "k",
        "orange",
        "purple",
        "brown",
        "lime",
        "pink",
        "navy",
        "teal",
        "coral",
        "olive",
        "chocolate",
        "indigo",
        "crimson",
        "gold",
    ]

    coords = np.array(coordinates)

    if demands is None:
        demands = {i: 0 for i in range(len(coords))}

    customer_demands = [demands.get(i, 0) for i in range(1, len(coords))]

    if len(customer_demands) > 0 and max(customer_demands) > 0:
        min_size = 50
        max_size = 300

        if max(customer_demands) == min(customer_demands):
            normalized_sizes = [100] * len(customer_demands)
        else:

            normalized_sizes = [
                min_size
                + (max_size - min_size)
                * (d - min(customer_demands))
                / (max(customer_demands) - min(customer_demands))
                for d in customer_demands
            ]
    else:

        normalized_sizes = [50] * len(coords[1:])

    # 仓库坐标
    plt.scatter(coords[0][0], coords[0][1], c="red", s=200, marker="*", label="仓库")

    # 客户节点（大小根据需求量调整）
    plt.scatter(coords[1:, 0], coords[1:, 1], c="blue", s=normalized_sizes)

    # 添加客户节点标签
    for i in range(1, len(coords)):
        plt.text(
            coords[i][0] + 0.5,
            coords[i][1] + 0.5,
            f"{i}({demands.get(i, 0)})",
            fontsize=10,
        )

    # 结果路径
    for i, route in enumerate(selected_routes):
        route_coords = coords[route]
        color = colors[i % len(colors)]
        plt.plot(
            route_coords[:, 0],
            route_coords[:, 1],
            c=color,
            linewidth=2,
            label=f"Vehichle {i+1}",
        )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.title("CVRP Solution")
    if data_source == "generate dataset":
        plt.xlim(0, 100)
        plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图表
    # plt.savefig(output_file)
    # plt.show()
