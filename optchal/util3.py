import json
import numpy as np
from itertools import permutations
import random
import time
import pprint
import sys

import matplotlib.pyplot as plt


# Change history
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

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
        return f'Rider([{self.type}, {self.speed}, {self.capa}, {self.var_cost}, {self.fixed_cost}, {self.service_time}, {self.available_number}])'

    # 주어진 거리에 대한 배달원 비용 계산
    # = 배달원별 고정비 + 이동거리로 계산된 변동비
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

    def __repr__(self) -> str:
        return f'Bundle(all_orders, {self.rider.type}, {self.shop_seq}, {self.dlv_seq}, {self.total_volume}, {self.feasible})'


# 주문들의 총 부피 계산
# shop_seq는 주문들의 pickup list
# Note: shop_seq는 주문 id의 list와 동일
def get_total_volume(all_orders, shop_seq):
    return sum(all_orders[k].volume for k in shop_seq)


# shop_seq의 순서로 pickup하고 dlv_seq 순서로 배달할 때 총 거리 계산
# Note: shop_seq 와 dlv_seq는 같은 주문 id들을 가져야 함. 즉, set(shop_seq) == seq(dlv_seq). (주문 id들의 순서는 바뀔 수 있음)
def get_total_distance(K, dist_mat, shop_seq, dlv_seq):
    return sum(dist_mat[i, j] for (i, j) in zip(shop_seq[:-1], shop_seq[1:])) + dist_mat[
        shop_seq[-1], dlv_seq[0] + K] + sum(dist_mat[i + K, j + K] for (i, j) in zip(dlv_seq[:-1], dlv_seq[1:]))


# shop_seq의 순서로 pickup하고 dlv_seq 순서로 배달할 때 pickup과 delivery시간을 반환
# Note: shop_seq 와 dlv_seq는 같은 주문 id들을 가져야 함. 즉, set(shop_seq) == seq(dlv_seq). (주문 id들의 순서는 바뀔 수 있음)
def get_pd_times(all_orders, rider, shop_seq, dlv_seq):
    K = len(all_orders)

    pickup_times = {}

    k = shop_seq[0]
    t = all_orders[k].order_time + all_orders[k].cook_time  # order time + order cook time
    pickup_times[k] = t
    for next_k in shop_seq[1:]:
        t = max(t + rider.T[k, next_k], all_orders[next_k].ready_time)  # max{travel time + service time, ready time}
        pickup_times[next_k] = t

        k = next_k

    dlv_times = {}

    k = dlv_seq[0]
    t += rider.T[shop_seq[-1], k + K]

    dlv_times[k] = t

    for next_k in dlv_seq[1:]:
        t += rider.T[k + K, next_k + K]

        dlv_times[next_k] = t

        k = next_k

    return pickup_times, dlv_times


# shop_seq의 순서로 pickup하고 dlv_seq 순서로 배달원 rider가 배달할 때 묶음주문 제약 만족 여부 테스트
# 모든 제약을 만족하면 0 반환
# 용량 제약을 만족하지 못하면 -1 반환
# 시간 제약을 만족하지 못하면 -2 반환
# Note: shop_seq 와 dlv_seq는 같은 주문 id들을 가져야 함. 즉, set(shop_seq) == seq(dlv_seq). (주문 id들의 순서는 바뀔 수 있음)
def test_route_feasibility(all_orders, rider, shop_seq, dlv_seq):
    total_vol = get_total_volume(all_orders, shop_seq)
    if total_vol > rider.capa:
        # Capacity overflow!
        return -1  # Capacity infeasibility

    pickup_times, dlv_times = get_pd_times(all_orders, rider, shop_seq, dlv_seq)

    for k, dlv_time in dlv_times.items():
        if dlv_time > all_orders[k].deadline:
            return -2  # Deadline infeasibility

    return 0


# 두 개의 bundle이 제약을 만족하면서 묶일 수 있는지 테스트
# 합쳐진 붂음배송 경로는 가능한 모든 pickup/delivery 조합을 확인
# 두 개의 bundle을 합치는게 가능하면 합쳐진 새로운 bundle을 반환
# 합치는게 불가능하면 None을 반환
# Note: 이 때 배달원은 두 개의 주어진 bundle을 배달하는 배달원들만 후보로 테스트(주어진 bundle에 사용되지 않는 배달원을 묶는게 가능할 수 있음!)
# Note: 여러개의 배달원으로 묶는게 가능할 때 가장 먼저 가능한 배달원 기준으로 반환(비용을 고려하지 않음)
def try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2):
    merged_orders = bundle1.shop_seq + bundle2.shop_seq
    total_volume = get_total_volume(all_orders, merged_orders)

    if bundle1.rider.type == bundle2.rider.type:
        riders = [bundle1.rider]
    else:
        riders = [bundle1.rider, bundle2.rider]

    for rider in riders:
        # We skip the test if there are too many orders
        if total_volume <= rider.capa and len(merged_orders) <= 5:
            for shop_pem in permutations(merged_orders):
                for dlv_pem in permutations(merged_orders):
                    feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                    if feasibility_check == 0:  # feasible!
                        total_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)
                        return Bundle(all_orders, rider, list(shop_pem), list(dlv_pem),
                                      bundle1.total_volume + bundle2.total_volume, total_dist)

    return None

# 주어진 bundle의 배달원을 변경하는것이 가능한지 테스트
# Note: 원래 bindle의 방문 순서가 최적이 아닐수도 있기 때문에 방문 순서 조합을 다시 확인함
def try_bundle_rider_changing(all_orders, bundle, rider):
    if bundle.rider.type != rider.type and bundle.total_volume <= rider.capa:
        orders = bundle.shop_seq
        for shop_pem in permutations(orders):
            for dlv_pem in permutations(orders):
                feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                if feasibility_check == 0:  # feasible!
                    # Note: in-place replacing!
                    bundle.shop_seq = list(shop_pem)
                    bundle.dlv_seq = list(dlv_pem)
                    bundle.rider = rider
                    bundle.update_cost()
                    return True

    return False


# 남아 있는 배달원 중에 *변동비*가 더 싼 배달원을 반환
# 더 싼 배달원이 없으면 None 반환
def get_cheaper_available_riders(all_riders, rider):
    for r in all_riders:
        if r.available_number > 0 and r.var_cost < rider.var_cost:
            return r

    return None


# 주어진 bundle list에서 임의로 두 개를 반환(중복없이)
def select_two_bundles(all_bundles):
    bundle1, bundle2 = random.sample(all_bundles, 2)
    return bundle1, bundle2


# 평균 비용(목적함수) 계산
# = 총 비용 / 주문 수
def get_avg_cost(all_orders, all_bundles):
    return sum([bundle.cost for bundle in all_bundles]) / len(all_orders)


# 주어진 bundle list에서 제출용 solution 포멧으로 반환
def create_solution(prob_name, bundles):
    sol = {
        'bundles': [
            # rider type, shop_seq, dlv_seq
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in bundles
        ]
    }
    return sol


# 주어진 solution의 feasibility를 테스트
# Note: solution은 [배달원유형, pickup 순서, 배달 순서]의 list
# 반환하는 dict에는 solution이 feasible일 경우에는 평균 비용등의 정보가 추가적으로 포함됨
# solution이 infeasible일 경우에는 그 이유가 'infeasibility' 항목(key)으로 반환
def solution_check(K, all_orders, all_riders, dist_mat, solution):
    total_cost = 0
    total_dist = 0

    infeasibility = None

    if isinstance(solution, list):

        used_riders = {
            'CAR': 0,
            'WALK': 0,
            'BIKE': 0
        }

        all_deliveies = []

        for bundle_info in solution:
            if not isinstance(bundle_info, list) or len(bundle_info) != 3:
                infeasibility = f'A bundle information must be a list of rider type, shop_seq, and dlv_seq! ===> {bundle_info}'
                break

            rider_type = bundle_info[0]
            shop_seq = bundle_info[1]
            dlv_seq = bundle_info[2]

            # rider type check
            if not rider_type in ['BIKE', 'WALK', 'CAR']:
                infeasibility = f'Rider type must be either of BIKE, WALK, or CAR! ===> {rider_type}'
                break

            # Get rider object
            rider = None
            for r in all_riders:
                if r.type == rider_type:
                    rider = r

            # Increase used rider by 1
            used_riders[rider_type] += 1

            # Pickup sequence check
            if not isinstance(shop_seq, list):
                infeasibility = f'The second bundle infomation must be a list of pickups! ===> {shop_seq}'
                break

            for k in shop_seq:
                if not isinstance(k, int) or k < 0 or k >= K:
                    infeasibility = f'Pickup sequence has invalid order number: {k}'
                    break

            # Delivery sequence check
            if not isinstance(dlv_seq, list):
                infeasibility = f'The third bundle infomation must be a list of deliveries! ===> {dlv_seq}'
                break

            for k in dlv_seq:
                if not isinstance(k, int) or k < 0 or k >= K:
                    infeasibility = f'Delivery sequence has invalid order number: {k}'
                    break

            # Volume check
            total_volume = get_total_volume(all_orders, shop_seq)
            if total_volume > rider.capa:
                infeasibility = f"Bundle's total volume exceeds the rider's capacity!: {total_volume} > {rider.capa}"
                break

            # Deadline chaeck
            pickup_times, dlv_times = get_pd_times(all_orders, rider.T, shop_seq, dlv_seq)
            for k in dlv_seq:
                all_deliveies.append(k)
                if dlv_times[k] > all_orders[k].deadline:
                    infeasibility = f'Order {k} deadline is violated!: {dlv_times[k]} > {all_orders[k].deadline}'
                    break

            dist = get_total_distance(K, dist_mat, shop_seq, dlv_seq)
            cost = rider.calculate_cost(dist)

            total_dist += dist
            total_cost += cost

            if infeasibility is not None:
                break

        if infeasibility is None:
            # Check used number of riders
            for r in all_riders:
                if r.available_number < used_riders[r.type]:
                    infeasibility = f'The number of used riders of type {r.type} exceeds the given available limit!'
                    break

            # Check deliveries
            for k in range(K):
                count = 0
                for k_sol in all_deliveies:
                    if k == k_sol:
                        count += 1

                if count > 1:
                    infeasibility = f'Order {k} is assigned more than once! ===> {count} > 1'
                    break
                elif count == 0:
                    infeasibility = f'Order {k} is NOT assigned!'
                    break

    else:
        infeasibility = 'Solution must be a list of bundle information!'

    if infeasibility is None:  # All checks are passed!
        checked_solution = {
            'total_cost': float(total_cost),
            'avg_cost': float(total_cost / K),
            'num_drivers': len(solution),
            'total_dist': int(total_dist),
            'feasible': True,
            'infeasibility': None,
            'bundles': solution
        }
    else:
        print(infeasibility)
        checked_solution = {
            'feasible': False,
            'infeasibility': infeasibility,
            'bundles': solution
        }

    return checked_solution


# 주어진 solution의 경로를 visualize
def draw_route_solution(all_orders, solution=None):
    plt.subplots(figsize=(8, 8))
    node_size = 5

    shop_x = [order.shop_lon for order in all_orders]
    shop_y = [order.shop_lat for order in all_orders]
    plt.scatter(shop_x, shop_y, c='red', s=node_size, label='SHOPS')

    dlv_x = [order.dlv_lon for order in all_orders]
    dlv_y = [order.dlv_lat for order in all_orders]
    plt.scatter(dlv_x, dlv_y, c='blue', s=node_size, label='DLVS')

    if solution is not None:

        rider_idx = {
            'BIKE': 0,
            'CAR': 0,
            'WALK': 0
        }

        for bundle_info in solution['bundles']:
            rider_type = bundle_info[0]
            shop_seq = bundle_info[1]
            dlv_seq = bundle_info[2]

            rider_idx[rider_type] += 1

            route_color = 'gray'
            if rider_type == 'BIKE':
                route_color = 'green'
            elif rider_type == 'WALK':
                route_color = 'orange'

            route_x = []
            route_y = []
            for i in shop_seq:
                route_x.append(all_orders[i].shop_lon)
                route_y.append(all_orders[i].shop_lat)

            for i in dlv_seq:
                route_x.append(all_orders[i].dlv_lon)
                route_y.append(all_orders[i].dlv_lat)

            plt.plot(route_x, route_y, c=route_color, linewidth=0.5)

    plt.legend()


# 주어진 soliution의 묶음 배송 방문 시간대를 visualize
def draw_bundle_solution(all_orders, all_riders, dist_mat, solution):
    plt.subplots(figsize=(6, len(solution['bundles'])))

    x_max = max([ord.deadline for ord in all_orders])

    bundle_gap = 0.3
    y = 0.2

    plt.yticks([])

    for idx, bundle_info in enumerate(solution['bundles']):
        rider_type = bundle_info[0]
        shop_seq = bundle_info[1]
        dlv_seq = bundle_info[2]

        rider = None
        for r in all_riders:
            if r.type == rider_type:
                rider = r

        y_delta = 0.2

        pickup_times, dlv_times = get_pd_times(all_orders, rider.T, shop_seq, dlv_seq)

        total_volume = 0
        for k in shop_seq:
            total_volume += all_orders[k].volume  # order volume
            plt.hlines(y + y_delta / 2, all_orders[k].ready_time, all_orders[k].deadline, colors='gray')
            plt.vlines(all_orders[k].ready_time, y, y + y_delta, colors='gray')
            plt.vlines(all_orders[k].deadline, y, y + y_delta, colors='gray')

            if total_volume > rider.capa:
                plt.scatter(pickup_times[k], y + y_delta / 2, c='red', zorder=100, marker='^', edgecolors='red',
                            linewidth=0.5)
            else:
                plt.scatter(pickup_times[k], y + y_delta / 2, c='green', zorder=100)

            if dlv_times[k] > all_orders[k].deadline:
                plt.scatter(dlv_times[k], y + y_delta / 2, c='red', zorder=100, marker='*', edgecolors='red',
                            linewidth=0.5)
            else:
                plt.scatter(dlv_times[k], y + y_delta / 2, c='orange', zorder=100)

            plt.text(all_orders[k].ready_time, y + y_delta / 2, f'{all_orders[k].ready_time} ', ha='right', va='center',
                     c='white', fontsize=6)
            plt.text(all_orders[k].deadline, y + y_delta / 2, f' {all_orders[k].deadline}', ha='left', va='center',
                     c='white', fontsize=6)

            y += y_delta

        dist = get_total_distance(len(all_orders), dist_mat, shop_seq, dlv_seq)
        cost = rider.calculate_cost(dist)

        plt.text(0, y + y_delta, f'{rider_type}: {shop_seq}-{dlv_seq}, tot_cost={cost}, tot_dist={dist}', ha='left',
                 va='top', c='gray', fontsize=8)
        y += bundle_gap
        plt.hlines(y, 0, x_max, colors='gray', linestyles='dotted')
        y += y_delta / 2

    plt.ylim(0, y)


def get_optimal_path(all_orders, bundle, rider):
    if bundle.rider.type != rider.type and bundle.total_volume <= rider.capa:
        orders = bundle.shop_seq
        is_feasible = False
        cur_optimal_cost = bundle.cost
        tmp_bundle = bundle.copy()
        ret_bundle = tmp_bundle
        for shop_pem in permutations(orders):
            for dlv_pem in permutations(orders):
                feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                if feasibility_check == 0:  # feasible!
                    # Note: in-place replacing!
                    tmp_bundle.shop_seq = list(shop_pem)
                    tmp_bundle.dlv_seq = list(dlv_pem)
                    tmp_bundle.rider = rider
                    tmp_bundle.update_cost()
                    is_feasible = True
                    if tmp_bundle.cost < cur_optimal_cost:
                        cur_optimal_cost = tmp_bundle.cost
                        ret_bundle = tmp_bundle.copy()
        if is_feasible:
            return ret_bundle
        else:
            return bundle


def get_cur_proba(E1, E2, k_, T):
    return np.exp((E1 - E2) / (k_ * T))

def select_three_bundles(all_bundles):
    bundle1, bundle2, bundle3 = random.sample(all_bundles, 3)
    return bundle1, bundle2, bundle3


