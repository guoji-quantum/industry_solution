import os
import time
import numpy as np
import json

from .dataset_generator import load_instance
from .route_generator import CVRPRouteGenerator
from .sa_solver import SimulatedAnnealingSolver
from .utils import plot_solution, plot_nodes, check_route_validation


class CVRPDataInterface:
    """
    CVRP数据接口类，用于存储和提供算法过程中的数据，供前端展示使用。
    """

    def __init__(self):
        # 问题实例数据
        self.instance_data = {
            "n_nodes": 0,
            "n_vehicles": 0,
            "capacity": 0,
            "coordinates": [],
            "demands": {},
            "distance_matrix": [],
        }

        # 聚类数据
        self.clustering_data = {"clusters": [], "cluster_loads": []}

        # 候选路径数据
        self.route_data = {"method": "", "candidate_routes": [], "route_costs": []}

        # 建模数据
        self.model_data = {"qubo": {}, "h": {}, "J": {}, "feed_dict": {}}

        # 求解结果数据
        self.sa_solution_data = {
            "solution": {},
            "result": {},
            "total_cost": 0,
            "runtime": 0,
        }

        # 结果验证
        self.verify_solution_data = {
            "is_feasible": [],
            "content_details": [],
        }

    def to_json(self):
        """将数据转换为JSON格式"""

        def numpy_converter(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            else:
                return str(obj)

        return json.dumps(self, default=numpy_converter, ensure_ascii=False, indent=4)

    def save_to_file(self, filepath):
        """将数据保存到文件"""
        with open(filepath, "w") as f:
            f.write(self.to_json())


class CVRPSolver:
    """
    CVRP求解器主类
    """

    def __init__(self, filedir, filename):

        self.instance = load_instance(filedir, filename)
        self.data_source = filedir
        self.distance_matrix = self.instance["distance_matrix"]
        self.demands = {int(k): v for k, v in self.instance["demands"].items()}
        self.capacity = self.instance["capacity"]
        self.n_vehicles = self.instance["n_vehicles"]
        self.n_nodes = self.instance["n_nodes"]
        self.coordinates = self.instance.get("coordinates", None)

        self.route_generator = CVRPRouteGenerator(
            self.distance_matrix,
            self.demands,
            self.capacity,
            self.n_vehicles,
            self.coordinates,
        )

        # 数据接口
        self.data_interface = CVRPDataInterface()
        self.data_interface.instance_data = {
            "n_nodes": self.n_nodes,
            "n_vehicles": self.n_vehicles,
            "capacity": self.capacity,
            "coordinates": self.coordinates,
            "demands": self.demands,
            "distance_matrix": self.distance_matrix,
        }

    def generate_routes(self, method):
        """
        生成CVRP的候选路径。

        """
        print(f"生成 {self.n_nodes} 个节点的候选线路")
        candidate_routes, clusters = self.route_generator.generate_routes(method=method)
        print(f"生成 {len(candidate_routes)} 条候选线路")

        # 计算候选路径的成本
        route_costs = []
        for route in candidate_routes:
            cost = 0
            for i in range(len(route) - 1):
                cost += self.distance_matrix[route[i]][route[i + 1]]
            route_costs.append(cost)

        # 存储聚类和路径数据到数据接口
        self.data_interface.clustering_data = {
            "clusters": clusters,
            "cluster_loads": [
                sum(self.demands[node] for node in cluster) for cluster in clusters
            ],
        }
        self.data_interface.route_data = {
            "method": method,
            "candidate_routes": candidate_routes,
            "route_costs": route_costs,
        }

        return candidate_routes, clusters

    def solve_with_sa(self, candidate_routes=None):
        """
        使用模拟量子退火算法求解CVRP。
        """
        print("\n模拟量子退火求解中...")

        # start_time = time.time()

        if candidate_routes is None:
            candidate_routes = self.generate_routes()

        qubo, h, J, model, feed_dict = self.route_generator.path_based_construct_qubo(
            candidate_routes
        )

        print(
            f"基于path-based formulation进行问题建模，候选路径数量为{len(candidate_routes)}"
        )

        # Solve with SA
        sa_solver = SimulatedAnnealingSolver(qubo, model, feed_dict)
        solution, result, total_cost, runtime = sa_solver.solve_and_print(
            candidate_routes, self.distance_matrix, self.n_vehicles
        )

        self.data_interface.model_data = {
            "qubo": {f"{i},{j}": v for (i, j), v in qubo.items()},
            "h": h,
            "J": {f"{i},{j}": v for (i, j), v in J.items()},
            "feed_dict": feed_dict,
        }

        # 存储SA求解结果到数据接口
        self.data_interface.sa_solution_data = {
            "solution": solution,
            "result": result,
            "total_cost": total_cost,
            "runtime": runtime,
        }

        # compute_time = time.time() - start_time
        print(f"模拟量子退火计算耗时: {runtime:.2f} 毫秒")
        # print(f"计算总耗时: {compute_time* 1000:.2f} 毫秒")

        return solution, result, total_cost, runtime

    def plot_sa_solution(
        self,
        solution,
        candidate_routes,
        output_file="../results/sa_solution.png",
    ):
        """
        绘制模拟量子退火的结果可视化图
        """
        plot_solution(
            solution, candidate_routes, self.coordinates, self.demands, self.data_source
        )

    def plot_problems_nodes(self):
        plot_nodes(self.demands, self.coordinates)

    def verify_sa_solution(self, result):
        is_feasible, content_details = check_route_validation(
            result, self.demands, self.n_nodes, self.n_vehicles, self.capacity
        )
        self.data_interface.verify_solution_data = {
            "is_feasible": is_feasible,
            "content_details": content_details,
        }

    def run_sa(self):
        candidate_routes, clusters = self.generate_routes("greedy")
        sa_solution, sa_result, sa_total_cost, sa_runtime = self.solve_with_sa(
            candidate_routes
        )
        self.verify_sa_solution(sa_result)

    def export_data_to_json(self, filedir, filename):
        data_filename = f"{filename.rsplit('.', 1)[0]}_data.json"
        dir_path = os.path.join("results", filedir)
        file_path = os.path.join(dir_path, data_filename)
        if file_path:
            os.makedirs(dir_path, exist_ok=True)
            self.data_interface.save_to_file(file_path)

        return self.data_interface.to_json()


if __name__ == "__main__":
    # # 运行示例
    # filename = "E-n23-k3.vrp"
    # filedir = "CVRPLIB/E"
    # solver = CVRPSolver(filedir, filename)

    # # 运行模拟退火算法
    # solver.run_sa()

    # # 输出数据到json文件
    # data_json = solver.export_data_to_json(filedir, filename)
    # print(f"数据已导出到 {filename.rsplit('.', 1)[0]}_data.json")

    filedir = "CVRPLIB/P"
    dir_path = os.path.join("dataset", filedir)
    file_list = os.listdir(dir_path)
    dataset_list = [file for file in file_list if file.endswith(".vrp")]
    for filename in dataset_list:
        print(f"正在处理 {filename}")
        solver = CVRPSolver(filedir, filename)
        solver.run_sa()
        solver.export_data_to_json(filedir, filename)
        print(f"数据已导出到 {filename.rsplit('.', 1)[0]}_data.json")
