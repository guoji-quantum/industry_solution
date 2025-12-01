import neal
import numpy as np
from pyqubo import Model
from .utils import print_solution, verify_solution


class SimulatedAnnealingSolver:
    """
    使用模拟退火求解CVRP问题。
    """

    def __init__(self, qubo, model, feed_dict):
        self.qubo = qubo
        self.model = model
        self.feed_dict = feed_dict

    def solve(self, num_reads=1000, annealing_time=400, step_size=0.1, verbose=True):
        """
        使用模拟退火求解CVRP问题。
        """
        if verbose:
            print("模拟量子退火求解中...")

        sampler = neal.SimulatedAnnealingSampler()
        sa_response = sampler.sample_qubo(
            self.qubo,
            num_reads=num_reads,
            annealing_time=annealing_time,
            step_size=step_size,
        )

        # 计算运行时间
        preprocessing_ns = sa_response.info["timing"]["preprocessing_ns"] / 1e6
        sampling_ns = sa_response.info["timing"]["sampling_ns"] / 1e6
        postprocessing_ns = sa_response.info["timing"]["postprocessing_ns"] / 1e6
        total_time = preprocessing_ns + sampling_ns + postprocessing_ns

        # 提取有效样本
        sa_samples = []
        sa_energies = []

        for record in sa_response.record:
            sample = dict(zip(sa_response.variables, record[0]))
            decoded = self.model.decode_sample(
                sample, vartype="BINARY", feed_dict=self.feed_dict
            )
            broken = decoded.constraints(only_broken=True)

            if len(broken) == 0:
                sa_samples.append(decoded.sample)
                sa_energies.append(decoded.energy)

        if not sa_samples:
            if verbose:
                print("未找到有效解决方案。考虑增加num_reads或调整惩罚权重。")
            # 返回第一个样本作为后备，即使它不满足所有约束
            sample = self.model.decode_sample(
                sa_response.first.sample, vartype="BINARY", feed_dict=self.feed_dict
            )
            best_solution = sample.sample
            success = False
        else:
            if verbose:
                print(f"找到 {len(sa_samples)} 个合理的解")
            # 获取能量最小的解决方案
            best_index = sa_energies.index(min(sa_energies))
            best_solution = sa_samples[best_index]
            success = True

        return best_solution, success, total_time

    def solve_and_print(
        self,
        candidate_routes,
        distance_matrix,
        n_vehicles,
        verbose=True,
    ):
        """
        求解CVRP并打印解决方案。
        """
        solution, success, runtime = self.solve(verbose=verbose)
        bitstring = [0] * len(solution)
        for k, v in solution.items():
            index = int(k[2:-1])
            bitstring[index] = v
        # r_indices = [candidate_routes[int(k[2:-1])] for k, v in solution.items() if v == 1]
        # 验证解决方案
        customers = set(range(1, distance_matrix.shape[0]))
        is_valid = verify_solution(bitstring, candidate_routes, customers, n_vehicles)

        if verbose:
            if is_valid:
                print("解验证:成功")
            else:
                print("解验证:失败")

        # 打印解决方案详情
        result, total_cost = print_solution(
            bitstring, candidate_routes, distance_matrix
        )

        return solution, result, total_cost, runtime
