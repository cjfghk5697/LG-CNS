# https://github.com/cjfghk5697/LG-CNS/blob/han/Han/based%20siwoo/weight%20update%20function%20copy%202/han_SA_myalgorithm.py 주문 조합 한 번씩만 확인하는 코드의 탐색 가중치 수량 확인 - 속도 및 비용 개선 시도

import math
import multiprocessing as mp
import random
import time

import numpy as np
from util_0704 import *


def cosine_annealing_warm_restarts(T0, T_mult, epochs, current_epoch):
    """
    Compute the temperature using Cosine Annealing Warm Restarts.
    T0: Initial temperature.
    T_mult: Factor to increase the period.
    epochs: Number of epochs.
    current_epoch: Current epoch.
    """
    T_cur = current_epoch % epochs
    T_i = epochs * T_mult
    return T0 * 0.5 * (1 + np.cos(np.pi * T_cur / T_i))

def simulated_annealing_weights_with_cawr(K, all_orders, all_riders, dist_mat,init_availables, timelimit=60, seed=None, T0=1000, T_mult=2, min_T=1, weight_change_range=0.05, restart_period=50):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    ALL_ORDERS = all_orders
    ALL_RIDERS = all_riders
    DIST = dist_mat

    start_time = time.time()

    # Initial solution using existing method
    init_availables = [rider.available_number for rider in all_riders]
    weight1, weight2, weight3, weight4 =  1, -2, 1,0

    inf = float('inf')

    car_rider = [rider for rider in all_riders if rider.type == 'CAR'][0]

    init_availables = [rider.available_number for rider in all_riders]

    all_bundles = []
    processed_weight_comb_c = 0
    min_weight_comb = []

    for ord in ALL_ORDERS:
        new_bundle = Bundle(ALL_ORDERS, car_rider, [ord.id], [ord.id], ord.volume, DIST[ord.id, ord.id+K])
        car_rider.available_number -= 1
        all_bundles.append(new_bundle)

    order_comb_possibility = [[True] * K for _ in range(K)]
    for i in range(len(all_bundles)):
        for j in range(i + 1, len(all_bundles)):
            bundle1 = all_bundles[i]
            bundle2 = all_bundles[j]

            order_num1 = bundle1.shop_seq[0]
            order_num2 = bundle2.shop_seq[0]

            ip = try_merging_bundles_by_dist(K, DIST, ALL_ORDERS, ALL_RIDERS, bundle1, bundle2)

            if not ip:
                order_comb_possibility[order_num1][order_num2] = False
                order_comb_possibility[order_num2][order_num1] = False

    optimized_order_perms = [dict(), dict(), dict()] # optimized_order_perms[rider_i] = {orders_sorted: 최적 번들}

    min_init_cost = inf
    epoch=0
    

    while True:
        cur_time = time.time()

        if cur_time - start_time > 60:
            break

        # Update temperature using CAWR
        T = cosine_annealing_warm_restarts(T0, T_mult, restart_period, epoch)
        epoch += 1
        if T < min_T:
            break

        # temp_start_time = time.time()
        bundles, result_rider_availables, cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
                K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3, weight4, order_comb_possibility, optimized_order_perms)
        # temp_end_time = time.time()
        # temp_process_time = temp_end_time - temp_start_time
        # Decide whether to accept the new weights


        if cost < min_init_cost:
            min_init_cost = cost
            min_init_cost_bundles = bundles
            min_init_cost_rider_availables = result_rider_availables
            min_weight_comb = (weight1, weight2, weight3, weight4)
        if K <= 150:

            bundles, result_rider_availables, _ = get_init_bundle(K, all_riders, all_orders, DIST, init_availables, weight1, weight2, weight3, weight4, try_merging_bundles_by_dist_possibles_only, order_comb_possibility, optimized_order_perms, False, 4)
            for rider_i in range(3):
                ALL_RIDERS[rider_i].available_number = result_rider_availables[rider_i]

            bundles, result_rider_availables = reassign_riders(K, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, bundles)
            cost = sum((bundle.cost for bundle in bundles)) / K

            
            if cost < min_init_cost:
                min_init_cost = cost
                min_init_cost_bundles = bundles
                min_init_cost_rider_availables = result_rider_availables
                min_weight_comb = (weight1, weight2, weight3, weight4)
        # Create a neighboring weights
        processed_weight_comb_c += 2
        weight1, weight2, weight3, weight4 = (weight1, weight2, weight3, weight4) + np.random.uniform(-weight_change_range, weight_change_range, size=4)
        weight1, weight2, weight3, weight4 = np.clip((weight1, weight2, weight3, weight4), -10, 10)  # Clip weights to a reasonable range
        for rider_i in range(3):
            ALL_RIDERS[rider_i].available_number = init_availables[rider_i]
            
    return processed_weight_comb_c, min_init_cost, min_weight_comb

# Updating the parallel function to use the new simulated annealing function
def parallel_simulated_annealing_with_cawr(K, all_orders, all_riders, dist_mat, init_availables, timelimit=60):
    start_time = time.time()

    # Number of parallel processes
    num_processes = 4
    pool = mp.Pool(processes=num_processes)

    # Seeds and parameters for random number generator and Simulated Annealing
    seeds = [random.randint(0, 100000) for _ in range(num_processes)]
    T0_values = np.linspace(15000, 25000, num_processes)
    T_mult_values = np.linspace(1, 3, num_processes)
    min_T_values = np.linspace(1, 5, num_processes)
    weight_change_range =  [0.35, 0.5, 0.65, 0.8]# Reduced range for changing weights
    restart_period_values = np.linspace(30, 60, num_processes).astype(int)

    # Run simulated_annealing_weights_with_cawr in parallel with different parameters
    results = [pool.apply_async(simulated_annealing_weights_with_cawr, args=(
    K, all_orders, all_riders, dist_mat,init_availables, timelimit, seeds[i], T0_values[i], T_mult_values[i], min_T_values[i], weight_change_range[i], restart_period_values[i])) 
               for i in range(num_processes)]
    pool.close()
    pool.join()

    # Collect results
    solutions = [res.get() for res in results]
    # Find the best solution
    final_bundles = min(solutions, key=lambda x: x[1])


    return final_bundles

def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):
    return parallel_simulated_annealing_with_cawr(K, all_orders, all_riders, dist_mat, timelimit)