# https://github.com/cjfghk5697/LG-CNS/blob/han/Han/based%20siwoo/weight%20update%20function%20copy%202/han_SA_myalgorithm.py

import multiprocessing as mp
import random
import time

from util_0802_ch import *


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
    weight1, weight2, weight3 = 1, -1, 1
    all_bundles, _, min_cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
        K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3
    )

    current_weights = np.array([weight1, weight2, weight3])
    current_cost = min_cost
    best_weights = current_weights.copy()
    best_cost = min_cost

    epoch = 0

    iteration_count = 0  # To count the number of iterations

    while epoch>=1 and time.time() - start_time < timelimit:
        iteration_count += 1

        # Update temperature using CAWR
        T = cosine_annealing_warm_restarts(T0, T_mult, restart_period, epoch)
        epoch += 1
        if T < min_T:
            break

        # Create a neighboring weights
        new_weights = current_weights + np.random.uniform(-weight_change_range, weight_change_range, size=3)
        new_weights = np.clip(new_weights, -10, 10)  # Clip weights to a reasonable range

        new_bundles, _, new_cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
            K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, *new_weights
        )

        # Decide whether to accept the new weights
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
            current_weights = new_weights
            current_cost = new_cost
            if new_cost < best_cost:
                best_weights = new_weights
                best_cost = new_cost

    # Final bundles with best weights
    final_bundles, result_rider_availables, final_cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
        K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, *best_weights
    )



    return final_bundles, result_rider_availables

# Updating the parallel function to use the new simulated annealing function
def parallel_simulated_annealing_with_cawr(K, all_orders, all_riders, dist_mat, init_availables, timelimit=60):
    start_time = time.time()

    # Number of parallel processes
    num_processes = 4
    pool = mp.Pool(processes=num_processes)

    # Seeds and parameters for random number generator and Simulated Annealing
    seeds = [random.randint(0, 100000) for _ in range(num_processes)]
    T0_values = np.linspace(1500, 2500, num_processes)
    T_mult_values = np.linspace(1, 3, num_processes)
    min_T_values = np.linspace(1, 5, num_processes)
    weight_change_range =  [0.1, 0.133, 0.1666, 0.21]# Reduced range for changing weights
    restart_period_values = np.linspace(30, 60, num_processes).astype(int)

    # Run simulated_annealing_weights_with_cawr in parallel with different parameters
    results = [pool.apply_async(simulated_annealing_weights_with_cawr, args=(
    K, all_orders, all_riders, dist_mat,init_availables, timelimit, seeds[i], T0_values[i], T_mult_values[i], min_T_values[i], weight_change_range[i], restart_period_values[i])) 
               for i in range(num_processes)]

    # Collect results
    solutions = [res.get() for res in results]

    # Find the best solution
    final_bundles, final_rider = min(solutions, key=lambda x: x[1])

    pool.close()
    pool.join()

    return final_bundles, final_rider


def simulated_annealing(K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, is_allow_worse_case):
    #--------------------------------------------- SA init ---------------------------------------------#

    before_SA_cost = sum(bundle.cost for bundle in all_bundles) / K
    print("SA starts!!! cur time :", time.time() - start_time)
    
    hist = []
    T = 100000000
    delta = 0.99
    T_final = 0.0001
    cur_solution = deepcopy(all_bundles)
    cur_cost = before_SA_cost
    SA_iter_cnt = 0
    T_mulitiplier = 0.00000001
    max_iter_cnt = 100
    T_max = T
    T_min = 10000

    for cur_bundle in cur_solution:
        cur_bundle.update_centroid()
    
    #--------------------------------------------- SA init ---------------------------------------------#

    #--------------------------------------------- SA iter ---------------------------------------------#

    is_pre_decreased = False
    
    while True:
        if time.time() - start_time > timelimit - 5 or T <= T_final: 
            break    
        SA_iter_cnt += 1
        
        new_solution, new_cost = make_new_solution(
            car_rider, K, cur_solution, all_riders, all_orders, dist_mat, T, is_pre_decreased, init_availables)
        
        if new_cost < cur_cost:
            cur_solution = new_solution
            cur_cost = new_cost
            is_pre_decreased = True
        elif new_cost == cur_cost:
            is_pre_decreased = False
            continue
        elif new_cost > cur_cost:
            E_multiplier = 1000 / cur_cost
            p = math.exp((-(new_cost - cur_cost) * E_multiplier) / (T * T_mulitiplier))

            print(int(new_cost), int(cur_cost), int(T), 
                (-(new_cost - cur_cost) * E_multiplier) / (T * T_mulitiplier), p)
            if is_allow_worse_case * p > random.random():
                print("changed")
                cur_solution = new_solution
                cur_cost = new_cost
            is_pre_decreased = False
        
        T = get_nxt_T_with_cos_annealing(T, T_min, T_max, SA_iter_cnt, max_iter_cnt)
        hist.append(cur_cost)

    print("SA iter cnt :", SA_iter_cnt)
    
    after_SA_cost = sum(bundle.cost for bundle in cur_solution) / K
    
    final_solution = cur_solution
    if after_SA_cost >= before_SA_cost:
        final_solution = all_bundles
        print("SA didn't work!!!")

    for bundle in final_solution:
        # if len(bundle.shop_seq) > 3:
        #     continue
        make_path_optimal(K, dist_mat, bundle, all_orders, all_riders)  

    print(sum(bundle.cost for bundle in final_solution) / K)
    final_availables = [rider.available_number for rider in all_riders]
    reassign_riders(K, ALL_ORDERS, ALL_RIDERS, DIST, final_availables, final_solution)
    
    plt.plot(hist)

    return final_solution
    #--------------------------------------------- SA iter ---------------------------------------------#
    
    #--------------------------------------------- SA iter ---------------------------------------------#
    

def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):

    ALL_ORDERS = all_orders
    ALL_RIDERS = all_riders
    DIST = dist_mat

    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time).astype(int)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#
    inf = float('inf')

    init_availables = deepcopy([rider.available_number for rider in all_riders])

    car_rider = [rider for rider in all_riders if rider.type == 'CAR'][0]
    bike_rider = [rider for rider in all_riders if rider.type == 'BIKE'][0]
    walk_rider = [rider for rider in all_riders if rider.type == 'WALK'][0]

    init_availables = [rider.available_number for rider in all_riders]

    min_init_cost = inf
    min_init_cost_bundle = []
    min_init_cost_rider_availables = []

    # try_merging_bundles_by_dist // 모든 배달원과 모든 조합에서의 최소 거리를 보는 케이스
    # try_merging_bundles_by_dist_walk_prefered // 위와 동일하지만 우선적으로 도보 배달원을 할당하는 케이스
    # try_merging_bundles_by_cost // 번들 합칠 때 거리가 아닌 비용을 보는 케이스
    min_init_cost_bundles, min_init_cost_rider_availables =parallel_simulated_annealing_with_cawr(K, all_orders, all_riders, dist_mat, init_availables, 15)
    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = min_init_cost_rider_availables[rider_i]
    all_bundles = min_init_cost_bundles
    print("SA start")
    freeze_support()

    num_core = 4
    with Pool(num_core) as pool:
        result = pool.starmap(simulated_annealing, [[K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, 1],
                                                    [K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, 1],
                                                    [K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, 1],
                                                    [K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, -1],])
        
    cost_result = [sum(bundle.cost for bundle in cur_solution) / K for cur_solution in result]
    final_solution_idx = np.argmin(cost_result)
    final_solution = result[final_solution_idx]
    #------------- End of custom algorithm code--------------#

    solution = [
            # rider type, shop_seq, dlv_seq
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in final_solution
    ]

    return solution