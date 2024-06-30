
import json
import numpy as np
from itertools import permutations
import random
import time
import pprint
import copy

import matplotlib.pyplot as plt

def get_dist_by_coords(x1, y1, x2, y2):
    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * 125950

    return dist

# 주문 class
class Order:
    def __init__(self, order_info):
        # [ORD_ID, ORD_TIME, SHOP_LAT, SHOP_LON, DLV_LAT, DLV_LON, VOL, COOK_TIME, DLV_DEADLINE]
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

# 크루스칼 알고리즘 방식을 활용하여 번들별 초기 할당을 하는 함수
# bundle_merging_function으로 합친 번들을 반환하는 함수를 사용 가능함
def get_init_bundle(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, bundle_merging_function):
    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = init_availables[rider_i]

    def find(v):
        while v != parent[v]:
            parent[v] = parent[parent[v]]
            v = parent[v]

        return v

    def union(a, b, new_bundle):
        if a > b:
            a, b = b, a

        parent[b] = a
        all_bundles[a] = new_bundle

    for r in ALL_RIDERS:
        r.T = np.round(DIST/r.speed + r.service_time).astype(int)

    car_rider = ALL_RIDERS[2]

    all_bundles = []
    for ord in ALL_ORDERS:
        new_bundle = Bundle(ALL_ORDERS, car_rider, [ord.id], [ord.id], ord.volume, DIST[ord.id, ord.id+K])
        car_rider.available_number -= 1
        all_bundles.append(new_bundle)

    for i in range(len(all_bundles)):
        bundle = all_bundles[i]

        shop_seq = bundle.shop_seq

        xs_s_sum = 0
        ys_s_sum = 0

        xs_e_sum = 0
        ys_e_sum = 0

        readytimes_sum = 0
        deadlines_sum = 0

        shop_seq_len = len(shop_seq)

        for order_num in shop_seq:
            order = ALL_ORDERS[order_num]

            xs_s_sum += order.shop_lat
            ys_s_sum += order.shop_lon

            xs_e_sum += order.dlv_lat
            ys_e_sum += order.dlv_lon

            readytimes_sum += order.ready_time
            deadlines_sum += order.deadline

        xs_s_avg = xs_s_sum / shop_seq_len
        ys_s_avg = ys_s_sum / shop_seq_len

        xs_e_avg = xs_e_sum / shop_seq_len
        ys_e_avg = ys_e_sum / shop_seq_len

        readytimes_avg = readytimes_sum / shop_seq_len
        deadlines_avg = deadlines_sum / shop_seq_len

        avg_info = [xs_s_avg, ys_s_avg, xs_e_avg, ys_e_avg, readytimes_avg, deadlines_avg]

        bundle.avg_info = avg_info

    edges = []
    for i in range(len(all_bundles)):
        for j in range(i + 1, len(all_bundles)):
            avg_info1 = all_bundles[i].avg_info
            avg_info2 = all_bundles[j].avg_info

            sx1, sy1, ex1, ey1, r1, d1 = avg_info1
            sx2, sy2, ex2, ey2, r2, d2 = avg_info2

            r_diff = abs(r1 - r2)
            d_diff = abs(d1 - d2)

            start_end_diff = get_dist_by_coords((sx1 + sx2) / 2, (sy1 + sy2) / 2, (ex1 + ex2) / 2, (ey1 + ey2) / 2)

            dist1 = DIST[i][j]
            dist2 = DIST[i + K][j + K]

            diff_score = dist1 + dist2 + r_diff * weight1 + d_diff * weight1 + start_end_diff * weight2

            edges.append((i, j, diff_score))

    parent = list(range(K))
    edges.sort(key=lambda x: x[2])

    for bundle_num1, bundle_num2, diff_score in edges:
        rbn1, rbn2 = find(bundle_num1), find(bundle_num2)

        if rbn1 == rbn2:
            continue
        
        new_bundle = bundle_merging_function(K, DIST, ALL_ORDERS, ALL_RIDERS, all_bundles[rbn1], all_bundles[rbn2])
        if new_bundle is not None:
            all_bundles[rbn1].rider.available_number += 1
            all_bundles[rbn2].rider.available_number += 1
            new_bundle.rider.available_number -= 1

            union(rbn1, rbn2, new_bundle)

    parent = [find(v) for v in parent]

    result_bundles = [all_bundles[v] for v in set(parent)]
    result_availables = [rider.available_number for rider in ALL_RIDERS]

    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = init_availables[rider_i]

    return result_bundles, result_availables, sum((bundle.cost for bundle in result_bundles)) / K

# 모든 번들에 걸쳐 배달원을 좀 더 최적으로 재배치한 결과를 반환해주는 함수
def check_reassign_riders(K, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, all_bundles):
    inf = float('inf')

    rider_types = [rider.type for rider in ALL_RIDERS]
    car_rider_index = rider_types.index('CAR')
    walk_rider_index = rider_types.index('WALK')
    bike_rider_index = rider_types.index('BIKE')

    rider_availables = init_availables.copy()

    bundle_info = [[''] * 3 for _ in range(len(all_bundles))]
    cost_info = [[inf] * 3 for _ in range(len(all_bundles))]

    for rider_i in range(3):
        rider = ALL_RIDERS[rider_i]

        for bundle_i in range(len(all_bundles)):
            old_bundle = all_bundles[bundle_i]
            volume = old_bundle.total_volume

            checked = check_bundle_rider_changing2(ALL_ORDERS, all_bundles[bundle_i], rider)

            if checked:
                shop_seq, dlv_seq, _ = checked

                new_bundle = Bundle(ALL_ORDERS, rider, shop_seq, dlv_seq, volume, get_total_distance(K, DIST, shop_seq, dlv_seq))

                bundle_info[bundle_i][rider_i] = new_bundle
                cost_info[bundle_i][rider_i] = new_bundle.cost

    # 한 종류의 라이더밖에 할당이 안되는 경우 또는 CAR이 가장 비용이 낮은 경우에 우선 할당
    to_dels = []
    new_all_bundles = []
    for i in range(len(bundle_info)):
        cost0, cost1, cost2 = cost_info[i]

        if cost0 == cost1 == inf:
            new_all_bundles.append(bundle_info[i][2])
            rider_availables[2] -= 1
            to_dels.append(i)
        elif cost0 == cost2 == inf:
            new_all_bundles.append(bundle_info[i][1])
            rider_availables[1] -= 1
            to_dels.append(i)       
        elif cost1 == cost2 == inf:
            new_all_bundles.append(bundle_info[i][0])
            rider_availables[0] -= 1
            to_dels.append(i)
        elif cost_info[i][car_rider_index] == min(cost_info[i]):
            new_all_bundles.append(bundle_info[i][car_rider_index])
            rider_availables[car_rider_index] -= 1
            to_dels.append(i)       

    to_dels = set(to_dels)
    bundle_info = [bundle_info[i] for i in range(len(bundle_info)) if i not in to_dels]
    cost_info = [cost_info[i] for i in range(len(cost_info)) if i not in to_dels]

    # WALK가 비용이 가장 낮은 경우 우선 할당 - 만약 WALK 배달원이 부족한 경우 두 번째로 비용이 적게 드는 경우와의 차이가 가장 큰 번들부터 할당
    walk_cands = []
    for i in range(len(bundle_info)):
        cost_info_sorted = sorted(cost_info[i])
        if cost_info[i][walk_rider_index] == min(cost_info[i]):
            diff = cost_info_sorted[1] - cost_info[i][walk_rider_index]
            walk_cands.append((diff, i))

    walk_cands.sort(key=lambda x: -x[0])
    walk_rider_available = rider_availables[walk_rider_index]
    walk_cands = walk_cands[:walk_rider_available]

    walk_cands_indices = []
    for _, i in walk_cands:
        walk_cands_indices.append(i)

        new_all_bundles.append(bundle_info[i][walk_rider_index])
        rider_availables[walk_rider_index] -= 1  

    walk_cands_indices = set(walk_cands_indices)
    bundle_info = [bundle_info[i] for i in range(len(bundle_info)) if i not in walk_cands_indices]
    cost_info = [cost_info[i] for i in range(len(cost_info)) if i not in walk_cands_indices]

    # BIKE가 비용이 가장 낮은 경우 우선 할당 - 만약 BIKE 배달원이 부족한 경우 CAR 배달원하고의 비용 차이가 가장 큰 번들부터 할당
    bike_cands = []
    for i in range(len(bundle_info)):
        cost_info_sorted = sorted(cost_info[i])
        if cost_info[i][bike_rider_index] == min(cost_info[i]):
            diff = cost_info[i][car_rider_index] - cost_info[i][bike_rider_index]
            bike_cands.append((diff, i))

    bike_cands.sort(key=lambda x: -x[0])
    bike_rider_available = rider_availables[bike_rider_index]
    bike_cands = bike_cands[:bike_rider_available]

    bike_cands_indices = []
    for _, i in bike_cands:
        bike_cands_indices.append(i)

        new_all_bundles.append(bundle_info[i][bike_rider_index])
        rider_availables[bike_rider_index] -= 1  

    bike_cands_indices = set(bike_cands_indices)
    bundle_info = [bundle_info[i] for i in range(len(bundle_info)) if i not in bike_cands_indices]
    cost_info = [cost_info[i] for i in range(len(cost_info)) if i not in bike_cands_indices]

    # 남은 배달원 중 가장 적은 비용을 가진 배달원에 바로 할당
    to_dels = []
    for i in range(len(bundle_info)):
        costs = [(cost_info[i][j], rider_types[j]) for j in range(3)]
        costs.sort(key=lambda x: x[0])

        for cost, rider_type in costs:
            rider_index = rider_types.index(rider_type)
            if rider_availables[rider_index]:
                rider_availables[rider_index] -= 1
                new_all_bundles.append(bundle_info[i][rider_index])
                to_dels.append(i)

                break

    to_dels = set(to_dels)
    bundle_info = [bundle_info[i] for i in range(len(bundle_info)) if i not in to_dels]
    cost_info = [cost_info[i] for i in range(len(cost_info)) if i not in to_dels]

    assert len(bundle_info) == len(cost_info) == 0

    return new_all_bundles, rider_availables

# 주문들의 총 부피 계산
# shop_seq는 주문들의 pickup list
# Note: shop_seq는 주문 id의 list와 동일
def get_total_volume(all_orders, shop_seq):
    return sum(all_orders[k].volume for k in shop_seq)

# shop_seq의 순서로 pickup하고 dlv_seq 순서로 배달할 때 총 거리 계산
# Note: shop_seq 와 dlv_seq는 같은 주문 id들을 가져야 함. 즉, set(shop_seq) == seq(dlv_seq). (주문 id들의 순서는 바뀔 수 있음)
def get_total_distance(K, dist_mat, shop_seq, dlv_seq):
    return sum(dist_mat[i,j] for (i,j) in zip(shop_seq[:-1], shop_seq[1:])) + dist_mat[shop_seq[-1], dlv_seq[0]+K] + sum(dist_mat[i+K,j+K] for (i,j) in zip(dlv_seq[:-1], dlv_seq[1:]))

# shop_seq의 순서로 pickup하고 dlv_seq 순서로 배달할 때 pickup과 delivery시간을 반환
# Note: shop_seq 와 dlv_seq는 같은 주문 id들을 가져야 함. 즉, set(shop_seq) == seq(dlv_seq). (주문 id들의 순서는 바뀔 수 있음)
def get_pd_times(all_orders, rider, shop_seq, dlv_seq):
    
    K = len(all_orders)

    pickup_times = {}

    k = shop_seq[0]
    t = all_orders[k].order_time + all_orders[k].cook_time # order time + order cook time
    pickup_times[k] = t
    for next_k in shop_seq[1:]:
        t = max(t+rider.T[k, next_k], all_orders[next_k].ready_time) # max{travel time + service time, ready time}
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
        return -1 # Capacity infeasibility

    pickup_times, dlv_times = get_pd_times(all_orders, rider, shop_seq, dlv_seq)

    for k, dlv_time in dlv_times.items():
        if dlv_time > all_orders[k].deadline:
            return -2 # Deadline infeasibility
    
    return 0

# 모든 배달원과 모든 조합에서 최소 거리에 해당하는 병합된 번들을 반환해줌. (util2의 try_merging_bundles4와 동일)
def try_merging_bundles_by_dist(K, dist_mat, all_orders, all_riders, bundle1, bundle2):
    merged_orders = bundle1.shop_seq + bundle2.shop_seq
    
    if len(merged_orders) >= 4:
        return None
    
    bundle1.rider.available_number += 1
    bundle2.rider.available_number += 1

    total_volume = get_total_volume(all_orders, merged_orders)

    inf = float('inf')

    min_total_dist_rider = ''
    min_total_dist_shop_pen = []
    min_total_dist_dlv_pen = []
    min_total_dist = inf
    for rider in all_riders:
        if rider.available_number == 0:
            continue

        if total_volume <= rider.capa:
            for shop_pem in permutations(merged_orders):
                for dlv_pem in permutations(merged_orders):
                    feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                    if feasibility_check == 0: # feasible!
                        total_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)

                        if total_dist < min_total_dist:
                            min_total_dist = total_dist
                            min_total_dist_shop_pen = shop_pem
                            min_total_dist_dlv_pen = dlv_pem
                            min_total_dist_rider = rider

    bundle1.rider.available_number -= 1
    bundle2.rider.available_number -= 1

    if min_total_dist != inf:
        return Bundle(all_orders, min_total_dist_rider, list(min_total_dist_shop_pen), list(min_total_dist_dlv_pen), bundle1.total_volume+bundle2.total_volume, min_total_dist)
    else:     
        return None

# 거리를 고려하여 배달원을 선정할 수 있게 변경
# 이중 permutation 및 target에 해당하는 라이더 종류 케이스 전부 확인 후 병합
# 도보 배달원을 우선시해서 배달
def try_merging_bundles_by_dist_walk_prefered(K, dist_mat, all_orders, all_riders, bundle1, bundle2):
    merged_orders = bundle1.shop_seq + bundle2.shop_seq
    
    if len(merged_orders) >= 4:
        return None
    
    bundle1.rider.available_number += 1
    bundle2.rider.available_number += 1

    total_volume = get_total_volume(all_orders, merged_orders)

    inf = float('inf')

    min_total_dist_shop_pen = []
    min_total_dist_dlv_pen = []
    min_total_dist = inf
    walk_rider = [rider for rider in all_riders if rider.type == 'WALK'][0]
    if walk_rider.available_number >= 1:
        if total_volume <= walk_rider.capa:
            for shop_pem in permutations(merged_orders):
                for dlv_pem in permutations(merged_orders):
                    feasibility_check = test_route_feasibility(all_orders, walk_rider, shop_pem, dlv_pem)
                    if feasibility_check == 0: # feasible!
                        total_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)
                        if total_dist < min_total_dist:
                            min_total_dist = total_dist
                            min_total_dist_shop_pen = shop_pem
                            min_total_dist_dlv_pen = dlv_pem

    if min_total_dist != inf:
        return Bundle(all_orders, walk_rider, list(min_total_dist_shop_pen), list(min_total_dist_dlv_pen), bundle1.total_volume+bundle2.total_volume, min_total_dist)

    min_total_dist_rider = ''
    min_total_dist_shop_pen = []
    min_total_dist_dlv_pen = []
    min_total_dist = inf
    for rider in all_riders:
        # 이전에 이미 확인한 WALK 배달원 케이스는 바로 넘김
        if rider.type == 'WALK':
            continue
        if rider.available_number == 0:
            continue

        if total_volume <= rider.capa:
            for shop_pem in permutations(merged_orders):
                for dlv_pem in permutations(merged_orders):
                    feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                    if feasibility_check == 0: # feasible!
                        total_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)
                        if total_dist < min_total_dist:
                            min_total_dist = total_dist
                            min_total_dist_shop_pen = shop_pem
                            min_total_dist_dlv_pen = dlv_pem
                            min_total_dist_rider = rider

    bundle1.rider.available_number -= 1
    bundle2.rider.available_number -= 1

    if min_total_dist != inf:
        return Bundle(all_orders, min_total_dist_rider, list(min_total_dist_shop_pen), list(min_total_dist_dlv_pen), bundle1.total_volume+bundle2.total_volume, min_total_dist)
    else:     
        return None

# 비용을 고려하여 배달원을 선정할 수 있게 변경
# 이중 permutation 및 target에 해당하는 라이더 종류 케이스 전부 확인 후 병합
def try_merging_bundles_by_cost(K, dist_mat, all_orders, all_riders, bundle1, bundle2, target = ['WALK', 'BIKE', 'CAR']):
    merged_orders = bundle1.shop_seq + bundle2.shop_seq
    
    if len(merged_orders) >= 4:
        return None
    
    bundle1.rider.available_number += 1
    bundle2.rider.available_number += 1

    total_volume = get_total_volume(all_orders, merged_orders)

    inf = float('inf')

    min_cost = inf
    min_cost_rider = ''
    min_cost_shop_pen = []
    min_cost_dlv_pen = []
    min_cost_dist = inf
    for rider in all_riders:
        if rider.type not in target:
            continue
        if rider.available_number == 0:
            continue

        if total_volume <= rider.capa:
            for shop_pem in permutations(merged_orders):
                for dlv_pem in permutations(merged_orders):
                    feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                    if feasibility_check == 0: # feasible!
                        dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)
                        cost = rider.calculate_cost(dist)

                        if cost < min_cost:
                            min_cost = cost
                            min_cost_shop_pen = shop_pem
                            min_cost_dlv_pen = dlv_pem
                            min_cost_rider = rider
                            min_cost_dist = dist

    bundle1.rider.available_number -= 1
    bundle2.rider.available_number -= 1

    if min_cost != inf:
        return Bundle(all_orders, min_cost_rider, list(min_cost_shop_pen), list(min_cost_dlv_pen), bundle1.total_volume+bundle2.total_volume, min_cost_dist)
    else:     
        return None

# 주어진 bundle의 배달원을 변경하는것이 가능한지 테스트
# Note: 원래 bindle의 방문 순서가 최적이 아닐수도 있기 때문에 방문 순서 조합을 다시 확인함
def try_bundle_rider_changing(all_orders, bundle, rider):
    if bundle.rider.type != rider.type and bundle.total_volume <= rider.capa:
        orders = bundle.shop_seq
        for shop_pem in permutations(orders):
            for dlv_pem in permutations(orders):
                feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                if feasibility_check == 0: # feasible!
                    # Note: in-place replacing!
                    bundle.shop_seq = list(shop_pem)
                    bundle.dlv_seq = list(dlv_pem)
                    bundle.rider = rider
                    bundle.update_cost()
                    return True

    return False

# 배달원을 바꾸는 작업을 직접 하지 않음
def check_bundle_rider_changing(all_orders, bundle, rider):
    if bundle.rider.type != rider.type and bundle.total_volume <= rider.capa:
        orders = bundle.shop_seq
        for shop_pem in permutations(orders):
            for dlv_pem in permutations(orders):
                feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                if feasibility_check == 0: # feasible!
                    return list(shop_pem), list(dlv_pem), rider

    return False

# check_bundle_rider_changing에서 기존과 같은 배달원인 경우 기존의 번들 정보를 반환하게 수정
def check_bundle_rider_changing2(all_orders, bundle, rider):
    if bundle.rider.type == rider.type:
        return bundle.shop_seq, bundle.dlv_seq, rider
    
    if bundle.total_volume <= rider.capa:
        orders = bundle.shop_seq
        for shop_pem in permutations(orders):
            for dlv_pem in permutations(orders):
                feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                if feasibility_check == 0: # feasible!
                    return list(shop_pem), list(dlv_pem), rider

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
        'bundles' : [
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
                if not isinstance(k, int) or k<0 or k>=K:
                    infeasibility = f'Pickup sequence has invalid order number: {k}'
                    break

            # Delivery sequence check
            if not isinstance(dlv_seq, list):
                infeasibility = f'The third bundle infomation must be a list of deliveries! ===> {dlv_seq}'
                break

            for k in dlv_seq:
                if not isinstance(k, int) or k<0 or k>=K:
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


    if infeasibility is None: # All checks are passed!
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

# 주어진 solution의 경로를 visualize + 지점별 숫자 텍스트 표시
def draw_route_solution2(all_orders, solution=None):
    
    plt.subplots(figsize=(15, 15))
    node_size = 5

    shop_x = [order.shop_lon for order in all_orders]
    shop_y = [order.shop_lat for order in all_orders]
    plt.scatter(shop_x, shop_y, c='red', s=node_size, label='SHOPS')

    delta = 0.4
    for i in range(len(all_orders)):
        x = shop_x[i]
        y = shop_y[i]

        plt.text(x, y, str(i + 1), fontsize=6)

    # plt.text(all_orders[k].ready_time, y+y_delta/2, f'{all_orders[k].ready_time} ', ha='right', va='center', c='white', fontsize=6)

    dlv_x = [order.dlv_lon for order in all_orders]
    dlv_y = [order.dlv_lat for order in all_orders]
    plt.scatter(dlv_x, dlv_y, c='blue', s=node_size, label='DLVS')

    for i in range(len(all_orders)):
        x = dlv_x[i]
        y = dlv_y[i]

        plt.text(x, y, str(i + 1), fontsize=6)

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
            total_volume += all_orders[k].volume # order volume
            plt.hlines(y+y_delta/2, all_orders[k].ready_time, all_orders[k].deadline, colors='gray')
            plt.vlines(all_orders[k].ready_time, y, y+y_delta, colors='gray')
            plt.vlines(all_orders[k].deadline, y, y+y_delta, colors='gray')

            if total_volume > rider.capa:
                plt.scatter(pickup_times[k], y+y_delta/2, c='red', zorder=100, marker='^', edgecolors='red', linewidth=0.5)
            else:
                plt.scatter(pickup_times[k], y+y_delta/2, c='green', zorder=100)

            if dlv_times[k] > all_orders[k].deadline:
                plt.scatter(dlv_times[k], y+y_delta/2, c='red', zorder=100, marker='*', edgecolors='red', linewidth=0.5)
            else:
                plt.scatter(dlv_times[k], y+y_delta/2, c='orange', zorder=100)

            plt.text(all_orders[k].ready_time, y+y_delta/2, f'{all_orders[k].ready_time} ', ha='right', va='center', c='white', fontsize=6)
            plt.text(all_orders[k].deadline, y+y_delta/2, f' {all_orders[k].deadline}', ha='left', va='center', c='white', fontsize=6)

            y += y_delta

        dist = get_total_distance(len(all_orders), dist_mat, shop_seq, dlv_seq)
        cost = rider.calculate_cost(dist)

        plt.text(0, y+y_delta, f'{rider_type}: {shop_seq}-{dlv_seq}, tot_cost={cost}, tot_dist={dist}', ha='left', va='top', c='gray', fontsize=8)
        y += bundle_gap
        plt.hlines(y, 0, x_max, colors='gray', linestyles='dotted')
        y += y_delta/2


    plt.ylim(0, y)


