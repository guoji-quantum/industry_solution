import numpy as np
import json
import os
import random
from .dataloading import parse_cvrp_data


class CVRPDatasetGenerator:
    """
    生成具有不同节点数量的CVRP（带容量限制的车辆路径问题）数据集。
    """

    def __init__(self, base_dir=None):
        # 如果未提供base_dir，检查当前或父目录中是否存在dataset目录
        if base_dir is None:
            if os.path.exists("dataset/generate dataset"):
                self.base_dir = "dataset/generate dataset"
            elif os.path.exists("../dataset/generate dataset"):
                self.base_dir = "../dataset/generate dataset"
            else:
                self.base_dir = "dataset/generate dataset"  # 默认为当前目录
        else:
            self.base_dir = base_dir

        os.makedirs(self.base_dir, exist_ok=True)

    def generate_random_instance(
        self,
        n_nodes,
        n_vehicles,
        demand_range=(1000, 3000),
        grid_size=100,
        seed=None,
    ):
        """
        根据节点数和小车数生成随机CVRP实例。
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        coordinates = np.zeros((n_nodes, 2))
        # 将仓库固定在 (50, 50)
        coordinates[0] = [50.0, 50.0]
        # 为其他节点（客户）生成随机坐标，范围为 (0, 100)
        for i in range(1, n_nodes):
            coordinates[i] = np.round(np.random.rand(2) * grid_size, 2)

        # 计算欧几里得距离矩阵
        distance_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    distance_matrix[i, j] = np.sqrt(
                        np.sum((coordinates[i] - coordinates[j]) ** 2)
                    )

        # 将距离保留两位小数
        distance_matrix = np.round(distance_matrix, 2)

        # 生成随机需求（仓库需求为0）
        demands = {0: 0}  # 仓库没有需求
        for i in range(1, n_nodes):
            demands[i] = random.randint(demand_range[0], demand_range[1])

        # 计算适当的车辆容量（每辆车总需求的40-60%）
        total_demand = sum(demands.values())

        # 设置容量，使n_vehicles能够处理所有需求并有一些余量
        vehicle_capacity = int(total_demand / (n_vehicles * 0.85))
        # 将容量调整为100的倍数
        vehicle_capacity = ((vehicle_capacity + 99) // 100) * 100

        # 创建实例数据
        instance = {
            "distance_matrix": distance_matrix.tolist(),
            "demands": demands,
            "capacity": vehicle_capacity,
            "n_vehicles": n_vehicles,
            "coordinates": coordinates.tolist(),
            "n_nodes": n_nodes,
        }

        return instance

    def generate_and_save_dataset(
        self,
        n_nodes,
        n_vehicles,
        seed_base=42,
    ):
        """
        生成并保存CVRP数据集。
        """
        if n_vehicles > np.ceil(np.sqrt(n_nodes))+1:
            raise ValueError(f"输入的n_nodes={n_nodes}和n_vehicles={n_vehicles}不满足要求。车辆数量不能超过节点数量的平方根向上取整值({np.ceil(np.sqrt(n_nodes))+1})")
        elif n_vehicles < 0 or n_nodes < 0:
            raise ValueError(f"输入的n_nodes={n_nodes}和n_vehicles={n_vehicles}不满足要求。车辆数量或节点数量不能小于0")
        elif n_vehicles > n_nodes:
            raise ValueError(f"输入的n_nodes={n_nodes}和n_vehicles={n_vehicles}不满足要求。车辆数量不能超过节点数量({n_nodes})")
        
        seed = seed_base + n_nodes * 100
        instance = self.generate_random_instance(n_nodes, n_vehicles, seed=seed)

        # 保存实例
        filename = f"cvrp_nodes_{n_nodes}_vehicles_{n_vehicles}_instance.json"
        filepath = os.path.join(self.base_dir, filename)

        with open(filepath, "w") as f:
            json.dump(instance, f, indent=2)

        print(f"Generated {filepath}")
        return filename


def load_instance(filedir, filename):
    """
    从文件加载CVRP实例。
    """
    if filedir == "generate dataset":
        filepath = os.path.join("dataset", filedir, filename)
        with open(filepath, "r") as f:
            instance = json.load(f)
    elif filedir == "CVRPLIB":
        dataset_class = filename.split("-")[0]
        filepath = os.path.join("dataset", filedir, dataset_class, filename)
        data_json = parse_cvrp_data(filepath)
        instance = json.loads(data_json)

    instance["distance_matrix"] = np.array(instance["distance_matrix"])

    return instance


def list_available_instances(base_dir=None):
    """
    列出数据集目录中所有可用的CVRP实例。
    """
    # 如果未提供base_dir，检查当前或父目录中是否存在dataset目录
    if base_dir is None:
        if os.path.exists("dataset/generate dataset"):
            base_dir = "dataset/generate dataset"
        elif os.path.exists("../dataset/generate dataset"):
            base_dir = "../dataset/generate dataset"
        else:
            base_dir = "dataset/generate dataset"  # 默认为当前目录

    return [f for f in os.listdir(base_dir) if f.endswith(".json")]


if __name__ == "__main__":
    # 示例用法
    generator = CVRPDatasetGenerator()

    generator.generate_and_save_dataset(n_nodes=21, n_vehicles=4, seed_base=47)

    # 列出可用实例
    print("\n可用实例:")
    instances = list_available_instances()
    for instance in instances:
        print(f"- {instance}")
