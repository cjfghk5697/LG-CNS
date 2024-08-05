# 0804_4, 5, 6, 7 + 4번들부터는 walk 사용 안함 - 디버깅 완료

import json
import numpy as np
from itertools import permutations
import random
import time
import pprint
import copy

import matplotlib.pyplot as plt

#---------------- added ----------------#
import heapq
import math
from copy import deepcopy
from functools import cmp_to_key
from multiprocessing import Pool, freeze_support
#---------------- added ----------------#

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
# class Bundle:
#     def __init__(self, all_orders, rider, shop_seq, dlv_seq, total_volume, total_dist, feasible=True):
#         self.rider = rider
#         self.all_orders = all_orders
#         self.feasible = feasible
#         self.shop_seq = shop_seq
#         self.dlv_seq = dlv_seq
#         self.total_volume = total_volume
#         self.total_dist = total_dist
#         self.update_cost()

#     # 묶음 주문의 비용 update
#     def update_cost(self):
#         self.cost = self.rider.calculate_cost(self.total_dist)
#         self.cost_per_ord = self.cost / len(self.shop_seq)

#     def __repr__(self) -> str:
#         return f'Bundle(all_orders, {self.rider.type}, {self.shop_seq}, {self.dlv_seq}, {self.total_volume}, {self.feasible})'
    
class Bundle:
    def __init__(self, all_orders, rider, shop_seq, dlv_seq, total_volume, total_dist, feasible=True):
        self.rider = rider
        self.all_orders = all_orders
        self.feasible = feasible
        self.shop_seq = shop_seq
        self.dlv_seq = dlv_seq
        self.total_volume = total_volume
        self.total_dist = total_dist
        self.shp_centroid_lat = 0
        self.shp_centroid_lon = 0
        self.dlv_centroid_lat = 0
        self.dlv_centroid_lon = 0
        self.update_centroid()
        self.update_cost()

    # 묶음 주문의 비용 update
    def update_cost(self):
        self.cost = self.rider.calculate_cost(self.total_dist)
        self.cost_per_ord = self.cost / len(self.shop_seq)

    def update_centroid(self):
        self.shp_centroid_lat = 0
        self.shp_centroid_lon = 0
        self.dlv_centroid_lat = 0
        self.dlv_centroid_lon = 0
        sz = len(self.shop_seq)
        for cur_shp in self.shop_seq:
            self.shp_centroid_lat += self.all_orders[cur_shp].shop_lat
            self.shp_centroid_lon += self.all_orders[cur_shp].shop_lon
            self.dlv_centroid_lat += self.all_orders[cur_shp].dlv_lat
            self.dlv_centroid_lon += self.all_orders[cur_shp].dlv_lon
        self.shp_centroid_lat /= sz
        self.shp_centroid_lon /= sz
        self.dlv_centroid_lat /= sz
        self.dlv_centroid_lon /= sz

    def __repr__(self) -> str:
        return f'Bundle(all_orders, {self.rider.type}, {self.shop_seq}, {self.dlv_seq}, {self.total_volume}, {self.feasible})'

# 크루스칼 알고리즘 방식을 활용하여 번들별 초기 할당을 하는 함수
def kruskal_bundling(K, DIST, ALL_ORDERS, ALL_RIDERS, weight1, weight2, weight3, bundle_merging_function, order_count_upper_limit, avg_method, all_bundles, order_comb_possibility, optimized_order_perms, is_random_num_added):
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

        time_window_length_avg = (deadlines_sum - readytimes_sum) / shop_seq_len

        avg_info = [xs_s_avg, ys_s_avg, xs_e_avg, ys_e_avg, readytimes_avg, deadlines_avg, time_window_length_avg]

        bundle.avg_info = avg_info

    if is_random_num_added:
        weight1 += random.choice([-1, -0.5, 0, 0.5, 1])
        weight2 += random.choice([-1, -0.5, 0, 0.5, 1])
        weight3 += random.choice([-0.5, 0, 0.5])

    edges = []
    for i in range(len(all_bundles)):
        for j in range(i + 1, len(all_bundles)):
            shop_seq1 = all_bundles[i].shop_seq
            shop_seq2 = all_bundles[j].shop_seq

            if len(shop_seq1) + len(shop_seq2) > order_count_upper_limit:
                continue

            ip = True
            for order1_num in shop_seq1:
                for order2_num in shop_seq2:
                    if not order_comb_possibility[order1_num][order2_num]:
                        ip = False
                        break
                if not ip:
                    break

            if not ip:
                continue
            
            avg_info1 = all_bundles[i].avg_info
            avg_info2 = all_bundles[j].avg_info

            sx1, sy1, ex1, ey1, r1, d1, twl1 = avg_info1
            sx2, sy2, ex2, ey2, r2, d2, twl2 = avg_info2

            r_diff = abs(r1 - r2)
            d_diff = abs(d1 - d2)

            twl_avg = (twl1 * len(all_bundles[i].shop_seq) + twl2 * len(all_bundles[j].shop_seq)) / (len(all_bundles[i].shop_seq) + len(all_bundles[j].shop_seq))

            start_end_diff = get_dist_by_coords((sx1 + sx2) / 2, (sy1 + sy2) / 2, (ex1 + ex2) / 2, (ey1 + ey2) / 2)

            if avg_method == 'avg':
                dist1 = get_dist_by_coords(sx1, sy1, sx2, sy2)
                dist2 = get_dist_by_coords(ex1, ey1, ex2, ey2)
            elif avg_method == 'two_seq':
                dist1 = DIST[i][j]
                dist2 = DIST[i + K][j + K]
            elif avg_method == 'two':
                order_num1 = all_bundles[i].shop_seq[0]
                order_num2 = all_bundles[j].shop_seq[0]

                dist1 = DIST[order_num1][order_num2]
                dist2 = DIST[order_num1 + K][order_num2 + K]  
            else:
                assert False

            diff_score = dist1 + dist2 + r_diff * weight1 + d_diff * weight1 + start_end_diff * weight2 + twl_avg * weight3
            if is_random_num_added:
                diff_score += random.randint(0, 500)

            edges.append((i, j, diff_score))

    parent = list(range(len(all_bundles)))
    edges.sort(key=lambda x: x[2])

    # edges 내 랜덤 이동
    # edge_len = len(edges)
    # for _ in range(len(edges) * 5):
    #     index = random.randint(0, edge_len - 2)
    #     edges[index], edges[index + 1] = edges[index + 1], edges[index]

    for bundle_num1, bundle_num2, diff_score in edges:
        rbn1, rbn2 = find(bundle_num1), find(bundle_num2)

        if rbn1 == rbn2:
            continue

        new_bundle = bundle_merging_function(K, DIST, ALL_ORDERS, ALL_RIDERS, all_bundles[rbn1], all_bundles[rbn2], order_comb_possibility, optimized_order_perms, order_count_upper_limit)

        if new_bundle is not None:
            all_bundles[rbn1].rider.available_number += 1
            all_bundles[rbn2].rider.available_number += 1
            new_bundle.rider.available_number -= 1

            union(rbn1, rbn2, new_bundle)

    parent = [find(v) for v in parent]

    result_bundles = [all_bundles[v] for v in set(parent)]
    rider_availables = [rider.available_number for rider in ALL_RIDERS]

    return result_bundles, rider_availables

# bundle_merging_function으로 합친 번들을 반환하는 함수를 사용 가능함
def get_init_bundle(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3, bundle_merging_function,  order_comb_possibility, optimized_order_perms, is_random_num_added, order_count_upper_limit=3):
    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = init_availables[rider_i]

    for r in ALL_RIDERS:
        r.T = np.round(DIST/r.speed + r.service_time).astype(int)

    car_rider = [rider for rider in ALL_RIDERS if rider.type == 'CAR'][0]

    all_bundles = []
    for ord in ALL_ORDERS:
        new_bundle = Bundle(ALL_ORDERS, car_rider, [ord.id], [ord.id], ord.volume, DIST[ord.id, ord.id+K])
        car_rider.available_number -= 1
        all_bundles.append(new_bundle)

    all_bundles, rider_abailables = kruskal_bundling(K, DIST, ALL_ORDERS, ALL_RIDERS, weight1, weight2, weight3, try_merging_bundles_by_dist_possibles_only, order_count_upper_limit, 'two_seq', all_bundles, order_comb_possibility, optimized_order_perms, is_random_num_added)

    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = init_availables[rider_i]

    return all_bundles, rider_abailables, sum((bundle.cost for bundle in all_bundles)) / K

# 2 -> 4 -> 3 형태 위주로 try_merging_bundles_by_dist를 사용하여 번들을 차례대로 생성하는 함수
def get_init_bundle_4_order_bundle_prefered(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3, order_comb_possibility, optimized_order_perms, is_random_num_added):
    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = init_availables[rider_i]

    for r in ALL_RIDERS:
        r.T = np.round(DIST/r.speed + r.service_time).astype(int)

    car_rider = [rider for rider in ALL_RIDERS if rider.type == 'CAR'][0]

    all_bundles = []
    for ord in ALL_ORDERS:
        new_bundle = Bundle(ALL_ORDERS, car_rider, [ord.id], [ord.id], ord.volume, DIST[ord.id, ord.id+K])
        car_rider.available_number -= 1
        all_bundles.append(new_bundle)

    # 2개 주문 묶음 생성
    all_bundles, _ = kruskal_bundling(K, DIST, ALL_ORDERS, ALL_RIDERS, weight1, weight2, weight3, try_merging_bundles_by_dist_possibles_only, 2, 'two_seq', all_bundles, order_comb_possibility, optimized_order_perms, is_random_num_added)

    # 4개 주문 묶음 생성
    all_bundles, _ = kruskal_bundling(K, DIST, ALL_ORDERS, ALL_RIDERS, weight1, weight2, weight3, try_merging_bundles_by_dist_possibles_only, 4, 'avg', all_bundles, order_comb_possibility, optimized_order_perms, is_random_num_added)

    # 2개 이하 주문이 묶인 번들을 전부 푼 다음 다시 생성
    new_all_bundles = []
    for bundle in all_bundles:
        if len(bundle.shop_seq) >= 3:
            new_all_bundles.append(bundle)
        else:
            old_rider = bundle.rider
            old_rider.available_number += 1
            for order_num in bundle.shop_seq:
                order = ALL_ORDERS[order_num]

                new_bundle = Bundle(ALL_ORDERS, car_rider, [order.id], [order.id], order.volume, DIST[order.id, order.id + K])
                car_rider.available_number -= 1
                new_all_bundles.append(new_bundle)

    result_bundles, result_availables = kruskal_bundling(K, DIST, ALL_ORDERS, ALL_RIDERS, weight1, weight2, weight3, try_merging_bundles_by_dist_possibles_only, 3, 'two', new_all_bundles, order_comb_possibility, optimized_order_perms, is_random_num_added)

    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = init_availables[rider_i]
    return result_bundles, result_availables, sum((bundle.cost for bundle in result_bundles)) / K

# 5 -> 2 -> 4 -> 3 형태 위주로 try_merging_bundles_by_dist를 사용하여 번들을 차례대로 생성하는 함수
def get_init_bundle_5_order_bundle_prefered(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3, order_comb_possibility, optimized_order_perms, is_random_num_added):
    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = init_availables[rider_i]

    for r in ALL_RIDERS:
        r.T = np.round(DIST/r.speed + r.service_time).astype(int)

    car_rider = [rider for rider in ALL_RIDERS if rider.type == 'CAR'][0]

    all_bundles = []
    for ord in ALL_ORDERS:
        new_bundle = Bundle(ALL_ORDERS, car_rider, [ord.id], [ord.id], ord.volume, DIST[ord.id, ord.id+K])
        car_rider.available_number -= 1
        all_bundles.append(new_bundle)

    # 최대 5개 주문 묶음 생성
    all_bundles, rider_availables = kruskal_bundling(K, DIST, ALL_ORDERS, ALL_RIDERS, weight1, weight2, weight3, try_merging_bundles_by_dist_possibles_only, 5, 'two_seq', all_bundles, order_comb_possibility, optimized_order_perms, is_random_num_added)
    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = rider_availables[rider_i]

    # 4, 5주문 번들은 따로 저장하고 3개 이하 주문이 묶인 번들을 전부 풀기
    grouped_bundles = []
    remained_bundles = []
    for bundle in all_bundles:
        if len(bundle.shop_seq) >= 4:
            grouped_bundles.append(bundle)
        else:
            old_rider = bundle.rider
            old_rider.available_number += 1
            for order_num in bundle.shop_seq:
                order = ALL_ORDERS[order_num]

                new_bundle = Bundle(ALL_ORDERS, car_rider, [order.id], [order.id], order.volume, DIST[order.id, order.id + K])
                car_rider.available_number -= 1
                remained_bundles.append(new_bundle)

    # 최대 2개 주문 묶음 생성
    remained_bundles, _ = kruskal_bundling(K, DIST, ALL_ORDERS, ALL_RIDERS, weight1, weight2, weight3, try_merging_bundles_by_dist_possibles_only, 2, 'two', remained_bundles, order_comb_possibility, optimized_order_perms, is_random_num_added)

    # 최대 4개 주문 묶음 생성
    remained_bundles, _= kruskal_bundling(K, DIST, ALL_ORDERS, ALL_RIDERS, weight1, weight2, weight3, try_merging_bundles_by_dist_possibles_only, 3, 'two', remained_bundles, order_comb_possibility, optimized_order_perms, is_random_num_added)
    
    # 3, 4주문 번들은 따로 저장하고 2개 이하 주문이 묶인 번들을 전부 풀기
    new_remained_bundles = []
    for bundle in remained_bundles:
        if len(bundle.shop_seq) >= 3:
            grouped_bundles.append(bundle)
        else:
            old_rider = bundle.rider
            old_rider.available_number += 1
            for order_num in bundle.shop_seq:
                order = ALL_ORDERS[order_num]

                new_bundle = Bundle(ALL_ORDERS, car_rider, [order.id], [order.id], order.volume, DIST[order.id, order.id + K])
                car_rider.available_number -= 1
                new_remained_bundles.append(new_bundle)

    # 최대 3개 주문 묶음 생성
    new_remained_bundles, result_rider_availables = kruskal_bundling(K, DIST, ALL_ORDERS, ALL_RIDERS, weight1, weight2, weight3, try_merging_bundles_by_dist_possibles_only, 3, 'two', new_remained_bundles, order_comb_possibility, optimized_order_perms, is_random_num_added)

    result_bundles = grouped_bundles + new_remained_bundles

    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = init_availables[rider_i]
    return result_bundles, result_rider_availables, sum((bundle.cost for bundle in result_bundles)) / K


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

def reassign_riders(K, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, all_bundles):
    ## ----------------- 기본 배달원 재할당 코드 -------------------
    for bundle in all_bundles:
        new_rider = get_cheaper_available_riders(ALL_RIDERS, bundle.rider)
        if new_rider is not None:
            old_rider = bundle.rider

            check_result = check_bundle_rider_changing(ALL_ORDERS, bundle, new_rider)
            if check_result:
                bundle.shop_seq = check_result[0]
                bundle.dlv_seq = check_result[1]
                bundle.rider = check_result[2]
                bundle.update_cost()

                old_rider.available_number += 1
                new_rider.available_number -= 1

    ## ----------------- 전체 배달원을 통째로 재할당하는 코드 -----------------------
    all_bundles, rider_availables = check_reassign_riders(K, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, all_bundles)

    return all_bundles, rider_availables

def reduce_dimension(sorted_orders):
    cur_w = 1

    reduced_value = 0
    for v in sorted_orders:
        to_add = (v + 1) * cur_w
        reduced_value += to_add

        cur_w *= 500

    return reduced_value

def get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3, order_comb_possibility, optimized_order_perms, is_random_num_added):
    all_bundles, rider_availables, _ = get_init_bundle_4_order_bundle_prefered(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3, order_comb_possibility, optimized_order_perms, is_random_num_added)
    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = rider_availables[rider_i]

    result_bundles, rider_availables = reassign_riders(K, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, all_bundles)
    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = init_availables[rider_i]
    
    return result_bundles, rider_availables, sum((bundle.cost for bundle in result_bundles)) / K

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

# 모든 배달원과 모든 조합에서 최소 거리에 해당하는 병합된 번들을 반환해줌.
def try_merging_bundles_by_dist(K, dist_mat, all_orders, all_riders, bundle1, bundle2, order_count_upper_limit=3, order_count_lower_limit=1):
    merged_orders = bundle1.shop_seq + bundle2.shop_seq
    
    if len(merged_orders) > order_count_upper_limit:
        return None
    if len(merged_orders) < order_count_lower_limit:
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


# 모든 배달원과 모든 조합에서 최소 거리에 해당하는 병합된 번들을 반환해줌. 이때 가능한 주문 쌍에 대한 정보를 참고하여 최적화를 진행함.
def try_merging_bundles_by_dist_possibles_only(K, dist_mat, all_orders, all_riders, bundle1, bundle2, order_comb_possibility, optimized_order_perms, order_count_upper_limit=3, order_count_lower_limit=1):
    shop_seq1 = bundle1.shop_seq
    shop_seq2 = bundle2.shop_seq

    for order1_num in shop_seq1:
        for order2_num in shop_seq2:
            if not order_comb_possibility[order1_num][order2_num]:
                return None

    merged_orders = shop_seq1 + shop_seq2
    merged_orders_sorted = tuple(sorted(merged_orders))

    if len(merged_orders) > order_count_upper_limit:
        return None
    if len(merged_orders) < order_count_lower_limit:
        return None
    
    bundle1.rider.available_number += 1
    bundle2.rider.available_number += 1

    if merged_orders_sorted in optimized_order_perms[0]:
        cands = [optimized_order_perms[rider_i][merged_orders_sorted] for rider_i in range(3)]
        cands = [Bundle(all_orders, all_riders[rider_i], v[0], v[1], v[2], v[3]) for rider_i, v in enumerate(cands) if v is not None and all_riders[rider_i].available_number >= 1]

        bundle1.rider.available_number -= 1
        bundle2.rider.available_number -= 1

        if not cands:
            return None
        else:
            cands.sort(key=lambda x: x.total_dist)

            selected = cands[0]

            return selected

    total_volume = get_total_volume(all_orders, merged_orders)

    inf = float('inf')

    min_total_dist_rider = ''
    min_total_dist_shop_pen = []
    min_total_dist_dlv_pen = []
    min_total_dist = inf
    for rider_i in range(3):
        rider = all_riders[rider_i]

        if rider.type == 'WALK' and len(merged_orders) >= 4:
            optimized_order_perms[rider_i][merged_orders_sorted] = None
            continue

        # if rider.available_number == 0:
        #     continue

        if total_volume <= rider.capa:
            rider_min_total_dist_shop_pen = []
            rider_min_total_dist_dlv_pen = []
            rider_min_total_dist = inf        
            for shop_pem in permutations(merged_orders):
                for dlv_pem in permutations(merged_orders):
                    feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                    if feasibility_check == 0: # feasible!
                        total_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)

                        if total_dist < rider_min_total_dist:
                            rider_min_total_dist = total_dist
                            rider_min_total_dist_shop_pen = shop_pem
                            rider_min_total_dist_dlv_pen = dlv_pem  

                        if  total_dist < min_total_dist and rider.available_number >= 1:
                            min_total_dist = total_dist
                            min_total_dist_shop_pen = shop_pem
                            min_total_dist_dlv_pen = dlv_pem
                            min_total_dist_rider = rider
            
            if rider_min_total_dist != inf:
                optimized_order_perms[rider_i][merged_orders_sorted] = (list(rider_min_total_dist_shop_pen), list(rider_min_total_dist_dlv_pen), bundle1.total_volume + bundle2.total_volume, rider_min_total_dist)
            else:
                optimized_order_perms[rider_i][merged_orders_sorted] = None
        else:
            optimized_order_perms[rider_i][merged_orders_sorted] = None

    bundle1.rider.available_number -= 1
    bundle2.rider.available_number -= 1

    if min_total_dist != inf:
        return Bundle(all_orders, min_total_dist_rider, list(min_total_dist_shop_pen), list(min_total_dist_dlv_pen), bundle1.total_volume + bundle2.total_volume, min_total_dist)
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

#------------------------------------------------ added ------------------------------------------------#
def get_cos_theta(sc1, dc1, sc2, dc2):
    cos_theta = 0
    v1 = [i - j for i, j in zip(sc1, dc1)]
    v2 = [i - j for i, j in zip(sc2, dc2)]

    cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]) * 
                                                   math.sqrt(v2[0] * v2[0] + v2[1] * v2[1]))
    return cos_theta


def get_cos_based_weight(all_orders, bundle1, bundle2):
    weight = 0

    shp_centroid1, shp_centroid2, dlv_centroid1, dlv_centroid2 = [0, 0], [0, 0], [0, 0], [0, 0]

    for i in bundle1.shop_seq:
        shp_centroid1[0] += all_orders[i].shop_lat
        shp_centroid1[1] += all_orders[i].shop_lon
        dlv_centroid1[0] += all_orders[i].dlv_lat
        dlv_centroid1[1] += all_orders[i].dlv_lon

    for i in bundle2.shop_seq:
        shp_centroid2[0] += all_orders[i].shop_lat
        shp_centroid2[1] += all_orders[i].shop_lon
        dlv_centroid2[0] += all_orders[i].dlv_lat
        dlv_centroid2[1] += all_orders[i].dlv_lon

    shp_centroid1 = [t / len(bundle1.shop_seq) for t in shp_centroid1]
    dlv_centroid1 = [t / len(bundle1.shop_seq) for t in dlv_centroid1]
    shp_centroid2 = [t / len(bundle2.shop_seq) for t in shp_centroid2]
    dlv_centroid2 = [t / len(bundle2.shop_seq) for t in dlv_centroid2]

    #cos_theta = get_cos_theta(shp_centroid1, dlv_centroid1, shp_centroid2, dlv_centroid2)
    cos_theta = 0
    dist = (math.sqrt((shp_centroid1[0] - shp_centroid2[0])**2 + (shp_centroid1[1] - shp_centroid2[1])**2) + 
            math.sqrt((dlv_centroid1[0] - dlv_centroid2[0])**2 + (dlv_centroid1[1] - dlv_centroid2[1])**2))
    weight = dist - 0 * cos_theta

    return weight


def get_dist_between_2centroids(bundle1, bundle2):
    #return 1
    return math.sqrt(math.sqrt((bundle1.shp_centroid_lat - bundle2.shp_centroid_lat)**2 + 
                      (bundle1.shp_centroid_lon - bundle2.shp_centroid_lon)**2) * 
            math.sqrt((bundle1.dlv_centroid_lat - bundle2.dlv_centroid_lat)**2 + 
                      (bundle1.dlv_centroid_lon - bundle2.dlv_centroid_lon)**2))


def SA_try_merging_bundles(K, dist_mat, all_orders, order_comb_possibility, bundle1, bundle2):
    shop_seq1 = bundle1.shop_seq
    shop_seq2 = bundle2.shop_seq

    for order1_num in shop_seq1:
        for order2_num in shop_seq2:
            if not order_comb_possibility[order1_num][order2_num]:
                return None

    merged_orders = shop_seq1 + shop_seq2
    merged_orders_shp = sorted(merged_orders, key = lambda x : all_orders[x].cook_time)
    merged_orders_dlv = sorted(merged_orders, key = lambda x : all_orders[x].deadline)
    
    total_volume = get_total_volume(all_orders, merged_orders)
    

    if bundle1.rider.type == bundle2.rider.type:
        riders = [bundle1.rider]
    else:
        riders = [bundle1.rider, bundle2.rider]

    for rider in riders:
        if rider.available_number <= 0: 
            continue
        # We skip the test if there are too many orders
        if total_volume <= rider.capa and len(merged_orders_shp) <= 4:
            for shop_pem in permutations(merged_orders_shp):
                for dlv_pem in permutations(merged_orders_dlv):
                    feasibility_check = test_route_feasibility(all_orders, rider, shop_pem, dlv_pem)
                    if feasibility_check == 0:  # feasible!
                        total_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)
                        return Bundle(all_orders, rider, list(shop_pem), list(dlv_pem),
                                      bundle1.total_volume + bundle2.total_volume, total_dist)
        #     print("time")
        # else:
        #     print("cap or volume")
    return None


# def insertion(cur_solution, cnt, all_orders, dist_mat, K):
#     for _ in range(cnt):
#         cur_bundle, nxt_bundle = random.sample(cur_solution, 2)
#         node = random.choice(cur_bundle.shop_seq)
#         tmp_bundle = Bundle(all_orders, nxt_bundle.rider, [node], [node], 
#                             all_orders[node].volume, dist_mat[node, node + K])
#         new_bundle = SA_try_merging_bundles(K, dist_mat, all_orders, bundle1=tmp_bundle, bundle2=nxt_bundle) 

#         if new_bundle is not None and new_bundle.rider.available_number > 0:
#             new_bundle.update_cost()
            
#             if len(cur_bundle.shop_seq) == 1:
#                 cur_bundle.rider.available_number += 1
#                 cur_solution.remove(cur_bundle)
                
#             else:
#                 cur_bundle.shop_seq.remove(node)
#                 cur_bundle.dlv_seq.remove(node)
#                 cur_bundle.total_volume = get_total_volume(all_orders, cur_bundle.shop_seq)
#                 cur_bundle.total_dist = get_total_distance(K, dist_mat, cur_bundle.shop_seq, cur_bundle.dlv_seq)
#                 cur_bundle.update_cost()

#             nxt_bundle.rider.available_number += 1
#             new_bundle.rider.available_number -= 1
#             cur_solution.remove(nxt_bundle)
#             cur_solution.append(new_bundle)


def insertion(cur_solution, cnt, all_orders, dist_mat, K, order_comb_possibility):
    for _ in range(cnt):
        # inverst_bundle_cnt = [math.sqrt(1 / len(cur_bundle.shop_seq)) for cur_bundle in cur_solution]
        # cur_bundle = random.choices(cur_solution, weights=inverst_bundle_cnt)[0]
        cur_bundle = random.choice(cur_solution)
        proba = []

        for cmp_bundle in cur_solution:
            cur_dist = get_dist_between_2centroids(cmp_bundle, cur_bundle)
            if cur_dist == 0: 
                cur_dist = 9999999999999999999999999999
            proba.append(1 / cur_dist)
            
        nxt_bundle = random.choices(cur_solution, weights=proba)[0]
        if nxt_bundle == cur_bundle:
            continue

        node = random.choice(cur_bundle.shop_seq)
        tmp_bundle = Bundle(all_orders, nxt_bundle.rider, [node], [node], 
                            all_orders[node].volume, dist_mat[node, node + K])
        new_bundle = SA_try_merging_bundles(K, dist_mat, all_orders, order_comb_possibility, bundle1=tmp_bundle, bundle2=nxt_bundle) 

        if new_bundle is not None and new_bundle.rider.available_number > 0:
            new_bundle.update_cost()
            
            if len(cur_bundle.shop_seq) == 1:
                cur_bundle.rider.available_number += 1
                cur_solution.remove(cur_bundle)
                
            else:
                cur_bundle.shop_seq.remove(node)
                cur_bundle.dlv_seq.remove(node)
                cur_bundle.total_volume = get_total_volume(all_orders, cur_bundle.shop_seq)
                cur_bundle.total_dist = get_total_distance(K, dist_mat, cur_bundle.shop_seq, cur_bundle.dlv_seq)
                cur_bundle.update_cost()
                cur_bundle.update_centroid()

            nxt_bundle.rider.available_number += 1
            new_bundle.rider.available_number -= 1
            new_bundle.update_centroid()
            cur_solution.remove(nxt_bundle)
            cur_solution.append(new_bundle)


def SA_get_cheaper_available_riders(all_riders, rider):
    for r in all_riders:
        if r.available_number > 0 and r.var_cost < rider.var_cost:
            return r

    return None


def mutation(cur_solution, cnt, all_orders, all_riders, car_rider):
    for _ in range(cnt):
        cur_bundle = random.choice(cur_solution)
        old_rider = cur_bundle.rider
        new_rider = SA_get_cheaper_available_riders(all_riders, old_rider)
        
        if new_rider is not None:
            if 0.2 > random.random():
                if try_bundle_rider_changing(all_orders, cur_bundle, car_rider):
                    old_rider.available_number += 1
                    new_rider.available_number -= 1
                
            if try_bundle_rider_changing(all_orders, cur_bundle, new_rider):
                old_rider.available_number += 1
                new_rider.available_number -= 1


def sort_by_bundle_cost(x, y):
    return y.cost - x.cost
    

# def rebundling(cur_solution, cnt, all_orders, car_rider, dist_mat, K):
#     for _ in range(cnt):
#         cur_bundle = random.choice(cur_solution)
#         old_rider = cur_bundle.rider
#         cur_shp_seq = cur_bundle.shop_seq
        
#         for i in cur_shp_seq:
#             new_bundle = Bundle(
#                 all_orders, car_rider, [i], [i], all_orders[i].volume, dist_mat[i, i+K])
#             cur_solution.append(new_bundle)
#             car_rider.available_number -= 1
        
#         cur_solution.remove(cur_bundle)
#         old_rider.available_number += 1
        
        
def rebundling(cur_solution, cnt, all_orders, car_rider, dist_mat, K):
    cur_solution = sorted(cur_solution, key=cmp_to_key(sort_by_bundle_cost))
    removed_bundles = []

    cnt = min(cnt, len(cur_solution) - 1)
    
    for i in range(cnt):
        cur_bundle = cur_solution[i]
        cur_bundle.rider.available_number += 1
        removed_bundles.append(cur_bundle)
    
    for _ in range(cnt):
        cur_bundle = cur_solution[0]
        cur_solution.remove(cur_bundle)
    
    for cur_bundle in removed_bundles:
        cur_shp_seq = cur_bundle.shop_seq
        
        for cur_shp in cur_shp_seq:
            new_bundle = Bundle(
                all_orders, car_rider, [cur_shp], [cur_shp], all_orders[cur_shp].volume, dist_mat[cur_shp, cur_shp+K])
            new_bundle.update_centroid()
            cur_solution.append(new_bundle)
            car_rider.available_number -= 1


def SA_test_route_feasibility(all_orders, rider, shop_seq, dlv_seq):
    total_vol = get_total_volume(all_orders, shop_seq)
    ret_dlv_time = 0
    
    if total_vol > rider.capa:
        # Capacity overflow!
        return -1, -1  # Capacity infeasibility

    pickup_times, dlv_times = get_pd_times(all_orders, rider, shop_seq, dlv_seq)

    for k, dlv_time in dlv_times.items():
        if dlv_time > all_orders[k].deadline:
            return -2, -1  # Deadline infeasibility
        else:
            ret_dlv_time = dlv_time
    return 0, ret_dlv_time


def make_path_optimal(K, dist_mat, cur_bundle, all_orders, all_riders):
    orders = deepcopy(cur_bundle.shop_seq)
    opt_cost = 1000000000000000
    for shop_pem in permutations(orders):
        for dlv_pem in permutations(orders):
            feasibility_check, dlv_time = SA_test_route_feasibility(all_orders, cur_bundle.rider, shop_pem, dlv_pem)
            if feasibility_check == 0:
                cur_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)
                if cur_dist < opt_cost:
                    opt_cost = cur_dist
                    cur_bundle.shop_seq = deepcopy(list(shop_pem))
                    cur_bundle.dlv_seq = deepcopy(list(dlv_pem))
                    cur_bundle.update_cost()
    
    old_rider = cur_bundle.rider
    new_rider = SA_get_cheaper_available_riders(all_riders, old_rider)
    if new_rider is not None:
        if try_bundle_rider_changing(all_orders, cur_bundle, new_rider):
                old_rider.available_number += 1
                new_rider.available_number -= 1
                               

def make_new_solution(car_rider, K, cur_solution, all_riders, all_orders, dist_mat, T, is_pre_decreased, init_availables, order_comb_possibility):
    new_solution = deepcopy(cur_solution)
    
    momentum = 1
    if not is_pre_decreased:
        momentum = 1
    
    insertion(new_solution, int(momentum * max(1, int(math.log2(T) * 1.2))), all_orders, dist_mat, K, order_comb_possibility)
    if 0.8 < random.random():
        mutation(new_solution, int(momentum * max(1, int(math.log2(T)))), all_orders, all_riders, car_rider)
    if 0.85 < random.random():
        rebundling(new_solution, int(momentum * max(1, int(math.log2(T) / 4))), all_orders, car_rider, dist_mat, K)
    # if 0.9 < random.random():
    #     reassign_riders(K, all_orders, all_riders, dist_mat, init_availables, cur_solution)
    
    new_cost = sum(bundle.cost for bundle in new_solution) / K
    return new_solution, new_cost


def get_nxt_T_with_cos_annealing(cur_T, T_min, T_max, tot_iter, max_iter):
    cur_iter = tot_iter 
    nxt_T = T_min + (T_max - T_min) * 0.5 * (1 + np.cos(np.pi * cur_iter / max_iter))
    
    return nxt_T

#------------------------------------------------ added ------------------------------------------------#