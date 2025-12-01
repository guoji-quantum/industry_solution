from networkx import center
import numpy as np
from itertools import permutations
from pyqubo import Binary, Constraint, Placeholder, Model, Array

from .utils import calculate_route_cost


class CVRPRouteGenerator:
    """
    使用聚类方法为CVRP生成候选路线。
    """

    def __init__(
        self,
        distance_matrix,
        demands,
        capacity,
        n_vehicles,
        coordinates,
    ):
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.capacity = capacity
        self.n_vehicles = n_vehicles
        self.n_nodes = distance_matrix.shape[0]
        self.n_customers = self.n_nodes - 1  # Exclude depot
        self.coordinates = coordinates

    def calculate_distance(self, node1, node2):
        node1_x, node1_y = self.coordinates[node1]
        node2_x, node2_y = self.coordinates[node2]
        cal_distance = np.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)
        return cal_distance

    def get_cluster_center(self, cluster):
        if len(cluster) == 1:
            return self.coordinates[cluster[0]]
        total_x = 0
        total_y = 0
        for i in cluster:
            center_x, center_y = self.coordinates[i]
            total_x += center_x
            total_y += center_y
        center_coordinates = [total_x / len(cluster), total_y / len(cluster)]
        return center_coordinates

    def get_cluster_center_index(self, cluster):
        center_coordinates = self.get_cluster_center(cluster)
        avg_dists = []
        for i in cluster:
            dist = np.sqrt(
                (center_coordinates[0] - self.coordinates[i][0]) ** 2
                + (center_coordinates[1] - self.coordinates[i][1]) ** 2
            )
            # dist = np.mean(
            #     [self.distance_matrix[i][j] for j in cluster if i != j]
            # )
            avg_dists.append((i, dist))
        center_index = min(avg_dists, key=lambda x: x[1])[0]
        return center_index

    def calculate_min_distance_to_cluster_centers(self, customer, clusters):
        """
        计算客户到所有现有聚类中心的最小距离
        """
        if not clusters or all(len(cluster) == 0 for cluster in clusters):
            # 如果没有现有聚类，返回到仓库的距离
            return self.calculate_distance(0, customer)
        
        min_distance = float('inf')
        for cluster in clusters:
            if len(cluster) > 0:
                center_coordinates = self.get_cluster_center(cluster)
                distance = np.sqrt(
                    (center_coordinates[0] - self.coordinates[customer][0]) ** 2
                    + (center_coordinates[1] - self.coordinates[customer][1]) ** 2
                )
                min_distance = min(min_distance, distance)
        
        return min_distance

    def cluster_generation(self, unassigned):
        """使用启发式聚类生成客户节点的聚类"""
        clusters = [[] for _ in range(self.n_vehicles)]
        cluster_loads = [0 for _ in range(self.n_vehicles)]
        for k in range(self.n_vehicles):
            if not unassigned:
                break

            # 选择核心停靠点：距离所有现有聚类中心最远的客户
            if k == 0:
                # 第一个聚类，选择距离仓库最远的客户
                core = max(unassigned, key=lambda i: self.calculate_distance(0, i))
            else:
                # 后续聚类，选择距离所有现有聚类中心最远的客户
                core = max(unassigned, key=lambda i: self.calculate_min_distance_to_cluster_centers(i, clusters[:k]))
            
            clusters[k].append(core)
            cluster_loads[k] += self.demands[core]
            unassigned.remove(core)

            # 不断添加最近的未分配客户，直到达到容量限制
            while True:
                if not unassigned:
                    break

                # 找到聚类中心：与聚类中其他节点平均距离最小的节点
                center_index = self.get_cluster_center_index(clusters[k])

                candidate_list = sorted(
                    list(unassigned),
                    key=lambda c: self.calculate_distance(center_index, c),
                )

                added = False
                for c in candidate_list:
                    if cluster_loads[k] + self.demands[c] <= self.capacity:
                        clusters[k].append(c)
                        cluster_loads[k] += self.demands[c]
                        unassigned.remove(c)
                        added = True
                        break
                if not added:
                    break

        return clusters, cluster_loads

    def cluster_adjusting(self, clusters, cluster_loads):
        """基于启发式聚类生成后的聚类记过，调整聚类"""
        changed = True
        while changed:
            changed = False
            for i in range(self.n_vehicles):
                for cust in list(clusters[i]):
                    # old_list = clusters[i].pop(cust)
                    current_center = self.get_cluster_center(clusters[i])
                    current_dist = np.sqrt(
                        (current_center[0] - self.coordinates[cust][0]) ** 2
                        + (current_center[1] - self.coordinates[cust][1]) ** 2
                    )

                    for j in range(self.n_vehicles):
                        if i == j:
                            continue
                        if cluster_loads[j] + self.demands[cust] > self.capacity:
                            continue
                        potential_center = self.get_cluster_center(clusters[j])
                        new_dist = np.sqrt(
                            (potential_center[0] - self.coordinates[cust][0]) ** 2
                            + (potential_center[1] - self.coordinates[cust][1]) ** 2
                        )
                        if new_dist < current_dist:
                            # 将客户移动到新的聚类
                            clusters[i].remove(cust)
                            cluster_loads[i] -= self.demands[cust]
                            clusters[j].append(cust)
                            cluster_loads[j] += self.demands[cust]
                            changed = True
                            break

                    if changed:
                        break
                if changed:
                    break

        return clusters

    def generate_all_routes_for_partition(self, partition):
        """
        为给定的客户分区生成所有可能的路线。
        """
        depot = 0  # 仓库索引
        routes = []

        for perm in permutations(partition):
            route = [depot] + list(perm) + [depot]
            routes.append(route)

        return routes

    def generate_route_set(self, partitions):
        """
        从客户分区生成一组候选路线。
        """
        all_possible_routes = []

        for partition in partitions:
            all_possible_routes.extend(
                self.generate_all_routes_for_partition(partition)
            )

        # 移除重复项（考虑两个方向）
        unique_routes = []
        route_set = set()
        for route in all_possible_routes:
            route_tuple = tuple(route)
            reversed_route_tuple = tuple(reversed(route))
            if route_tuple not in route_set and reversed_route_tuple not in route_set:
                unique_routes.append(route)
                route_set.add(route_tuple)

        return unique_routes

    def multi_start_greedy(self, partitions):
        """多起点贪心算法生成候选路线"""
        unique_routes = []
        route_set = set()
        for partition in partitions:
            for start in partition:
                path = [start]
                unvisited = set(partition)
                unvisited.remove(start)
                current = start
                while unvisited:
                    next_node = min(
                        unvisited,
                        key=lambda node: self.distance_matrix[current][node],
                    )
                    path.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node

                full_path = [0] + path + [0]
                route_tuple = tuple(full_path)
                reversed_route_tuple = tuple(reversed(full_path))
                if (
                    route_tuple not in route_set
                    and reversed_route_tuple not in route_set
                ):
                    unique_routes.append(full_path)
                    route_set.add(route_tuple)
        return unique_routes

    def validate_set_integrity(self, clusters):
        """验证聚类分簇是否成功"""
        nodes_set = set()
        integrity_nodes_set = set([i for i in range(self.n_nodes)])
        for cluster in clusters:
            for node in cluster:
                nodes_set.add(node)
        nodes_set.add(0)
        if nodes_set == integrity_nodes_set:
            print("聚类分簇成功！没有遗漏节点！")
        else:
            ignore_nodes = integrity_nodes_set - nodes_set
            print(f"聚类分簇失败！存在遗漏节点:{ignore_nodes}")

    def capacity_based_assignment(self, clusters, cluster_loads, unassigned_nodes):
        """
        基于容量的精细分配算法，仅考虑容量约束，不考虑距离因素。
        在现有聚类算法失败时使用，确保所有点都被分配且不超过容量限制。
        """
        if not unassigned_nodes:
            return clusters, cluster_loads

        print(f"执行基于容量的精细分配算法，处理 {len(unassigned_nodes)} 个未分配节点")

        # 按需求量从大到小排序未分配节点
        sorted_nodes = sorted(
            list(unassigned_nodes), key=lambda i: self.demands[i], reverse=True
        )

        # 尝试将未分配节点分配到现有簇中
        remaining_nodes = []
        for node in sorted_nodes:
            assigned = False
            # 尝试分配到有足够容量的簇中
            for k in range(self.n_vehicles):
                if cluster_loads[k] + self.demands[node] <= self.capacity:
                    clusters[k].append(node)
                    cluster_loads[k] += self.demands[node]
                    assigned = True
                    break

            if not assigned:
                remaining_nodes.append(node)

            # 只考虑 demands 和 capacity 约束，不考虑距离因素
            if remaining_nodes:
                print(
                    f"节点交换后仍有 {len(remaining_nodes)} 个节点未分配，尝试完全重构"
                )

                all_nodes = remaining_nodes.copy()
                for k in range(self.n_vehicles):
                    all_nodes.extend(clusters[k])
                    clusters[k] = []
                cluster_loads = [0] * self.n_vehicles

                # 按需求量从大到小排序所有节点
                all_nodes.sort(key=lambda i: self.demands[i], reverse=True)

                # 使用First-Fit Decreasing算法重新分配
                for node in all_nodes:
                    assigned = False
                    for k in range(self.n_vehicles):
                        if cluster_loads[k] + self.demands[node] <= self.capacity:
                            clusters[k].append(node)
                            cluster_loads[k] += self.demands[node]
                            assigned = True
                            break

                    if not assigned:
                        print(
                            f"警告：节点 {node}（需求量 {self.demands[node]}）无法分配！"
                        )
                        print(f"当前簇负载: {cluster_loads}")
                        print(f"总需求量: {sum(self.demands.values())}")
                        print(f"总容量: {self.capacity * self.n_vehicles}")
                        raise ValueError("无法找到满足容量约束的分配方案！")

        return clusters, cluster_loads

    def generate_routes(self, method):
        """
        使用聚类为CVRP生成候选路线。
        """
        customers = list(range(1, self.n_nodes))  # 客户索引：1到n-1（不包括仓库0）
        unassigned = set(customers)

        # 步骤1：聚类生成
        clusters, cluster_loads = self.cluster_generation(unassigned)

        # 步骤2：聚类调整
        clusters = self.cluster_adjusting(clusters, cluster_loads)

        # 验证聚类分簇是否成功
        self.validate_set_integrity(clusters)

        # 检查是否有未分配的节点
        nodes_set = set()
        for cluster in clusters:
            for node in cluster:
                nodes_set.add(node)

        integrity_nodes_set = set(range(1, self.n_nodes))  # 不包括仓库0
        unassigned_nodes = integrity_nodes_set - nodes_set

        # 如果有未分配节点，执行基于容量的精细分配
        if unassigned_nodes:
            print(f"发现 {len(unassigned_nodes)} 个未分配节点，执行基于容量的精细分配")
            clusters, cluster_loads = self.capacity_based_assignment(
                clusters, cluster_loads, unassigned_nodes
            )

            # 再次验证
            self.validate_set_integrity(clusters)

        # 步骤3：两种方法生成候选路线
        if method == "traversal":
            candidate_routes = self.generate_route_set(clusters)
        elif method == "greedy":
            candidate_routes = self.multi_start_greedy(clusters)

        return candidate_routes, clusters

    def path_based_construct_qubo(self, candidate_routes):
        """
        构建CVRP的QUBO模型。
        """
        n_routes = len(candidate_routes)

        x = Array.create("x", shape=(n_routes), vartype="BINARY")

        P = Placeholder("P")

        # 目标：最小化总路线成本
        objective = sum(
            x[i] * calculate_route_cost(candidate_routes[i], self.distance_matrix)
            for i in range(n_routes)
        )

        # 约束1：每个客户必须恰好被访问一次
        constraint1 = 0
        for i in range(1, self.n_nodes):  # Customer indices start from 1
            constraint1 += Constraint(
                (
                    sum(
                        x[idx]
                        for idx, route in enumerate(candidate_routes)
                        if i in route
                    )
                    - 1
                )
                ** 2,
                label=f"visit_customer_{i}",
            )

        # 约束2：使用的车辆数量不能超过限制
        constraint2 = Constraint(
            (sum(x[i] for i in range(n_routes)) - self.n_vehicles) ** 2,
            label="vehicles_limit",
        )

        H = objective + P * constraint1 + P * constraint2

        model = H.compile()

        feed_dict = {"P": 200.0}

        qubo, offset = model.to_qubo(feed_dict=feed_dict)
        h, J, offset = model.to_ising(feed_dict=feed_dict)

        return qubo, h, J, model, feed_dict
