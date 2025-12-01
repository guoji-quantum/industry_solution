import xml.etree.ElementTree as ET
import numpy as np
import json


def parse_cvrp_data(file_path):
    if file_path.endswith(".xml"):
        return parse_cvrp_xml_data(file_path)
    elif file_path.endswith(".vrp"):
        return parse_cvrp_vrp_data(file_path)
    elif file_path.endswith(".sol"):
        return parse_cvrp_sol_data(file_path)


def parse_cvrp_xml_data(file_path):
    """解析 .xml cvrp 数据"""
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 解析节点
    nodes = {}
    for node in root.find(".//nodes"):
        node_id = int(node.attrib["id"])
        cx = float(node.find("cx").text)
        cy = float(node.find("cy").text)
        nodes[node_id] = {"cx": cx, "cy": cy}

    # 解析车辆
    fleet_info = root.find(".//vehicle_profile")
    fleet = {
        "departure_node": int(fleet_info.find("departure_node").text),
        "arrival_node": int(fleet_info.find("arrival_node").text),
        "capacity": float(fleet_info.find("capacity").text),
    }

    # 解析 requests
    requests = {}
    for request in root.find(".//requests"):
        request_id = int(request.attrib["id"])
        node = int(request.attrib["node"])
        quantity = float(request.find("quantity").text)
        requests[request_id] = {"node": node, "quantity": quantity}

    return {
        "nodes": nodes,
        "fleet": fleet,
        "requests": requests,
    }


def parse_cvrp_vrp_data(file_path):
    """解析 .vrp cvrp 数据"""
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = {}
    node_coords = {}
    demands = {}
    depot = None
    section = None

    for line in lines:
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            section = "NODE_COORD_SECTION"
            continue
        elif line == "DEMAND_SECTION":
            section = "DEMAND_SECTION"
            continue
        elif line == "DEPOT_SECTION":
            section = "DEPOT_SECTION"
            continue
        elif line == "EOF":
            break

        if section == "NODE_COORD_SECTION":
            parts = line.split()
            node_id = int(parts[0])
            x_coord = float(parts[1])
            y_coord = float(parts[2])
            node_coords[node_id] = [x_coord, y_coord]
        elif section == "DEMAND_SECTION":
            parts = line.split()
            node_id = int(parts[0])
            demand = int(parts[1])
            demands[node_id - 1] = demand
        elif section == "DEPOT_SECTION":
            depot = int(line)
        else:
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

    data["n_nodes"] = int(data["NAME"].split("-")[1][1:])
    data["n_vehicles"] = int(data["NAME"].split("-")[2][1:])

    data["DIMENSION"] = int(data["DIMENSION"])
    data["capacity"] = int(data["CAPACITY"])

    coordinates = [None] * (data["DIMENSION"])
    for k, v in node_coords.items():
        coordinates[k - 1] = v

    data["coordinates"] = coordinates
    data["demands"] = demands
    data["depot"] = depot

    # 计算距离矩阵
    dimension = data["DIMENSION"]
    dist_matrix = np.zeros((dimension, dimension))
    for i in range(1, dimension + 1):
        for j in range(1, dimension + 1):
            if i != j:
                dist_matrix[i - 1][j - 1] = np.linalg.norm(
                    np.array(node_coords[i]) - np.array(node_coords[j])
                )
    data["distance_matrix"] = dist_matrix.tolist()

    return json.dumps(data, ensure_ascii=False, indent=4)


def parse_cvrp_sol_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    solution = {"routes": [], "cost": None}

    for line in lines:
        line = line.strip()
        if line.startswith("Route #"):
            # 解析结果路径
            route = list(map(int, line.split(":")[1].strip().split()))
            solution["routes"].append(route)
        elif line.startswith("Cost"):
            # 解析 cost
            solution["cost"] = int(line.split()[1])

    return solution
