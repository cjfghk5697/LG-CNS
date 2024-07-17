import json
import numpy as np
from itertools import permutations
import random
import time
import pprint
import matplotlib.pyplot as plt

# Change history
# 2024/7/1 - Update total_dist for a new bundle as well in try_bundle_rider_changing()
# 2024/6/21 - Fixed a comment in Order.__init__()
# 2024/6/16 - Fixed a bug that does not set the bundle routes in try_bundle_rider_changing()
# 2024/5/17 - Fixed a bug in get_pd_times()

# 주문 class
class Order:
    def __init__(self, order_info):
        # [ORD_ID, ORD_TIME, SHOP_LAT, SHOP_LON, DLV_LAT, DLV_LON, COOK_TIME, VOL, DLV_DEADLINE]
        self.id = order_info[0]
        self.order_time = order_info[1]
        self.shop_lat = order_info[2]
        self.shop_lon = order_info[3]
        self.dlv_lat = order_info[4]
        self.dlv_lon = order_info[5]
        self.cook_time = order_info[6]
        self.volume = order_info[7]
        self.deadline = order_info[8]
        self.ready_time = self.order_time + self.cook_time

    def __repr__(self):
        return f'Order([{self.id}, {self.order_time}, {self.shop_lat}, {self.shop_lon}, {self.dlv_lat}, {self.dlv_lon}, {self.volume}, {self.cook_time}, {self.deadline}])'

# 배달원 class
class Rider:
    def __init__(self, rider_info):
        # [type, speed, capa, var_cost, fixed_cost, service_time, available number]
        self.type = rider_info[0]
        self.speed = rider_info[1]
        self.capa = rider_info[2]
        self.var_cost = rider_info[3]
        self.fixed_cost = rider_info[4]
        self.service_time = rider_info[5]
        self.available_number = rider_info[6]
    
    def __repr__(self):
        return f'Rider([{self.type}, {self.speed}, {self.capa}, {self.var_cost}, {self.fixed_cost}, {self.service_time}, {self.available_number}])'
    
    # 주어진 거리에 대한 배달원 비용 계산
    def calculate_cost(self, dist):
        return self.fixed_cost + dist / 100.0 * self.var_cost

# 묶음 주문 정보
class Bundle:
    def __init__(self, all_orders, rider, shop_seq, dlv_seq, total_volume, total_dist, feasible=True):
        self.rider = rider
        self.all_orders = all_orders
        self.feasible = feasible
        self.shop_seq = shop_seq
        self.dlv_seq = dlv_seq
        self.total_volume = total_volume
        self.total_dist = total_dist
        self.update_cost()

    # 묶음 주문의 비용 update
    def update_cost(self):
        self.cost = self.rider.calculate_cost(self.total_dist)
        self.cost_per_ord = self.cost / len(self.shop_seq)

    def __repr__(self):
        return f'Bundle(all_orders, {self.rider.type}, {self.shop_seq}, {self.dlv_seq}, {self.total_volume}, {self.feasible})'

# 주문들의 총 부피 계산
def get_total_volume(all_orders, shop_seq):
    return sum(all_orders[k].volume for k in shop_seq)

# shop_seq의 순서로 pickup하고 dlv_seq 순서로 배달할 때 총 거리 계산
def get_total_distance(K, dist_mat, shop_seq, dlv_seq):
    return sum(dist_mat[i, j] for (i, j) in zip(shop_seq[:-1], shop_seq[1:])) + dist_mat[shop_seq[-1], dlv_seq[0] + K] + sum(dist_mat[i + K, j + K] for (i, j) in zip(dlv_seq[:-1], dlv_seq[1:]))

# shop_seq의 순서로 pickup하고 dlv_seq 순서로 배달할 때 pickup과 delivery시간을 반환
def get_pd_times(all_orders, rider, shop_seq, dlv_seq):
    K = len(all_orders)
    pickup_times = {}
    k = shop_seq[0]
    t = all_orders[k].order_time + all_orders[k].cook_time  # order time + order cook time
    pickup_times[k] = t
    for next_k in shop_seq[1:]:
        t = max(t + rider.service_time, all_orders[next_k].ready_time)  # max{travel time + service time, ready time}
        pickup_times[next_k] = t
        k = next_k
    dlv_times = {}
    k = dlv_seq[0]
    t += rider.service_time
    dlv_times[k] = t
    for next_k in dlv_seq[1:]:
        t += rider.service_time
        dlv_times[next_k] = t
        k = next_k
    return pickup_times, dlv_times

# shop_seq의 순서로 pickup하고 dlv_seq 순서로 배달원 rider가 배달할 때 묶음주문 제약 만족 여부 테스트
def test_route_feasibility(all_orders, rider, shop_seq, dlv_seq):
    total_vol = get_total_volume(all_orders, shop_seq)
    if total_vol > rider.capa:
        return -1  # Capacity infeasibility
    pickup_times, dlv_times = get_pd_times(all_orders, rider, shop_seq, dlv_seq)
    for k, dlv_time in dlv_times.items():
        if dlv_time > all_orders[k].deadline:
            return -2  # Deadline infeasibility
    return 0

# 두 개의 bundle이 제약을 만족하면서 묶일 수 있는지 테스트
def try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2):
    merged_orders = bundle1.shop_seq + bundle2.shop_seq
    total_volume = get_total_volume(all_orders, merged_orders)
    if bundle1.rider.type == bundle2.rider.type:
        riders = [bundle1.rider]
    else:
        riders = [bundle1.rider, bundle2.rider]
    for rider in riders:
        if total_volume <= rider.capa and len(merged_orders) <= 5:
            for shop_perm in permutations(merged_orders):
                for dlv_perm in permutations(merged_orders):
                    feasibility_check = test_route_feasibility(all_orders, rider, shop_perm, dlv_perm)
                    if feasibility_check == 0:  # feasible!
                        total_dist = get_total_distance(K, dist_mat, shop_perm, dlv_perm)
                        return Bundle(all_orders, rider, list(shop_perm), list(dlv_perm), bundle1.total_volume + bundle2.total_volume, total_dist)
    return None

# 주어진 bundle의 배달원을 변경하는 것이 가능한지 테스트
def try_bundle_rider_changing(all_orders, dist_mat, bundle, rider):
    if bundle.rider.type != rider.type and bundle.total_volume <= rider.capa:
        orders = bundle.shop_seq
        for shop_perm in permutations(orders):
            for dlv_perm in permutations(orders):
                feasibility_check = test_route_feasibility(all_orders, rider, shop_perm, dlv_perm)
                if feasibility_check == 0:  # feasible!
                    bundle.shop_seq = list(shop_perm)
                    bundle.dlv_seq = list(dlv_perm)
                    bundle.rider = rider
                    bundle.total_dist = get_total_distance(len(all_orders), dist_mat, bundle.shop_seq, bundle.dlv_seq)
                    bundle.update_cost()
                    return True
    return False

# 남아 있는 배달원 중에 변동비가 더 싼 배달원을 반환
def get_cheaper_available_riders(all_riders, rider):
    for r in all_riders:
        if r.available_number > 0 and r.var_cost < rider.var_cost:
            return r
    return None

# 주어진 bundle list에서 임의로 두 개를 반환(중복 없이)
def select_two_bundles(all_bundles):
    return random.sample(all_bundles, 2)

# 평균 비용(목적함수) 계산
def get_avg_cost(all_orders, all_bundles):
    return sum(bundle.cost for bundle in all_bundles) / len(all_orders)

# 주어진 bundle list에서 제출용 solution 포맷으로 반환
def create_solution(prob_name, bundles):
    return {
        'bundles': [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in bundles
        ]
    }

# 주어진 solution의 feasibility를 테스트
def solution_check(K, all_orders, all_riders, dist_mat, solution):
    total_cost = 0
    total_dist = 0
    infeasibility = None
    if isinstance(solution, list):
        used_riders = {'CAR': 0, 'WALK': 0, 'BIKE': 0}
        all_deliveries = []
        for bundle_info in solution:
            if not isinstance(bundle_info, list) or len(bundle_info) != 3:
                infeasibility = f'A bundle information must be a list of rider type, shop_seq, and dlv_seq!'
                break
            rider_type, shop_seq, dlv_seq = bundle_info
            if not isinstance(rider_type, str) or not isinstance(shop_seq, list) or not isinstance(dlv_seq, list):
                infeasibility = f'The information type of bundle is wrong! Must be (str, list, list).'
                break
            used_riders[rider_type] += 1
            # all_deliveries.extend(shop_seq)
            # all_deliveries.extend(dlv_seq)
            all_deliveries += shop_seq + dlv_seq
            # check whether this bundle is feasible
            matched_riders = [rider for rider in all_riders if rider.type == rider_type]
            if not matched_riders:
                infeasibility = f'The rider type {rider_type} does not exist!'
                break
            rider = matched_riders[0]
            total_vol = get_total_volume(all_orders, shop_seq)
            if total_vol > rider.capa:
                infeasibility = f'A bundle with {shop_seq} has a volume infeasibility!'
                break
            pickup_times, dlv_times = get_pd_times(all_orders, rider, shop_seq, dlv_seq)
            for k, dlv_time in dlv_times.items():
                if dlv_time > all_orders[k].deadline:
                    infeasibility = f'A bundle with {shop_seq} has a time infeasibility!'
                    break
            total_dist += get_total_distance(K, dist_mat, shop_seq, dlv_seq)
            total_cost += rider.calculate_cost(total_dist)
        if infeasibility is None:
            if len(set(all_deliveries)) < K * 2:
                infeasibility = f'There are missing or duplicated delivery points'
        if infeasibility is None:
            for rider_type, count in used_riders.items():
                matched_riders = [rider for rider in all_riders if rider.type == rider_type]
                if count > matched_riders[0].available_number:
                    infeasibility = f'The number of used riders is infeasible'
    else:
        infeasibility = 'Solution must be a list!'
    if infeasibility is None:
        return (total_cost, total_dist)
    else:
        return infeasibility

# TSP를 해결할 수 있는 솔루션 반환
def solve_tsp_using_ortools(distance_matrix):
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    size = len(distance_matrix)
    if size > 0:
        tsp_size = size
        num_routes = 1
        depot = 0
        routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        def distance_callback(from_index, to_index):
            return int(distance_matrix[from_index][to_index])
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            tsp_path = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                tsp_path.append(routing.IndexToNode(index))
                index = assignment.Value(routing.NextVar(index))
            tsp_path.append(routing.IndexToNode(index))
            return tsp_path
    return None

# 거리 matrix 생성
def get_dist_matrix(all_orders):
    K = len(all_orders)
    dist_mat = np.zeros((2 * K, 2 * K))
    for i in range(2 * K):
        for j in range(2 * K):
            if i < K and j < K:
                # 두 픽업 지점 사이 거리
                dist_mat[i, j] = np.hypot(all_orders[i].shop_lat - all_orders[j].shop_lat, all_orders[i].shop_lon - all_orders[j].shop_lon)
            elif i >= K and j >= K:
                # 두 배달 지점 사이 거리
                dist_mat[i, j] = np.hypot(all_orders[i - K].dlv_lat - all_orders[j - K].dlv_lat, all_orders[i - K].dlv_lon - all_orders[j - K].dlv_lon)
            elif i < K <= j:
                # 픽업 지점과 배달 지점 사이 거리
                dist_mat[i, j] = np.hypot(all_orders[i].shop_lat - all_orders[j - K].dlv_lat, all_orders[i].shop_lon - all_orders[j - K].dlv_lon)
            else:
                # 배달 지점과 픽업 지점 사이 거리
                dist_mat[i, j] = np.hypot(all_orders[i - K].dlv_lat - all_orders[j].shop_lat, all_orders[i - K].dlv_lon - all_orders[j].shop_lon)
    return dist_mat

# 비용 정보를 포함한 solution file 생성
def write_solution_file(filename, instance_name, solution, cost):
    solution_dict = {
        "instance": instance_name,
        "solution": solution,
        "cost": cost
    }
    with open(filename, 'w') as f:
        json.dump(solution_dict, f, indent=2)

# 묶음 solution을 개선하는 local search heuristic 함수
def local_search(prob_name, all_orders, all_riders, dist_mat, initial_solution, max_iterations=100):
    current_solution = initial_solution
    current_cost, _ = solution_check(len(all_orders), all_orders, all_riders, dist_mat, current_solution)
    best_solution = current_solution
    best_cost = current_cost
    for _ in range(max_iterations):
        bundle1, bundle2 = select_two_bundles(current_solution)
        merged_bundle = try_merging_bundles(len(all_orders), dist_mat, all_orders, bundle1, bundle2)
        if merged_bundle:
            new_solution = [b for b in current_solution if b != bundle1 and b != bundle2] + [merged_bundle]
            new_cost, _ = solution_check(len(all_orders), all_orders, all_riders, dist_mat, new_solution)
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
        for bundle in current_solution:
            cheaper_rider = get_cheaper_available_riders(all_riders, bundle.rider)
            if cheaper_rider:
                success = try_bundle_rider_changing(all_orders, dist_mat, bundle, cheaper_rider)
                if success:
                    new_solution = [b if b != bundle else bundle for b in current_solution]
                    new_cost, _ = solution_check(len(all_orders), all_orders, all_riders, dist_mat, new_solution)
                    if new_cost < best_cost:
                        best_solution = new_solution
                        best_cost = new_cost
        current_solution = best_solution
        current_cost = best_cost
    return best_solution, best_cost

def solve(prob_name, order_file, rider_file):
    with open(order_file, 'r') as f:
        all_orders = [Order(order_info) for order_info in json.load(f)]
    with open(rider_file, 'r') as f:
        all_riders = [Rider(rider_info) for rider_info in json.load(f)]
    dist_mat = get_dist_matrix(all_orders)
    initial_solution = []
    for order in all_orders:
        feasible_riders = [rider for rider in all_riders if order.volume <= rider.capa]
        if not feasible_riders:
            raise ValueError(f"No feasible rider found for order {order.id}")
        rider = min(feasible_riders, key=lambda r: r.var_cost)
        initial_solution.append(Bundle(all_orders, rider, [order.id], [order.id], order.volume, 0))
    best_solution, best_cost = local_search(prob_name, all_orders, all_riders, dist_mat, initial_solution)
    final_solution = create_solution(prob_name, best_solution)
    write_solution_file(f"{prob_name}_solution.json", prob_name, final_solution, best_cost)
    print(f"Best cost: {best_cost}")
    pprint.pprint(final_solution)
    