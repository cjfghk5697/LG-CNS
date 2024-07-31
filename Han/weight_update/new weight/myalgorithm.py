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

def simulated_annealing_weights_with_cawr(K, all_orders, all_riders, dist_mat, timelimit=60, seed=None, T0=1000, T_mult=2, min_T=1, weight_change_range=0.05, restart_period=50):
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

    while time.time() - start_time < timelimit:
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
    final_bundles, _, final_cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
        K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, *best_weights
    )

    solution = [
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in final_bundles
    ]

    return solution, best_cost, best_weights

# Updating the parallel function to use the new simulated annealing function
def parallel_simulated_annealing_with_cawr(K, all_orders, all_riders, dist_mat, timelimit=60):
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
    K, all_orders, all_riders, dist_mat, timelimit, seeds[i], T0_values[i], T_mult_values[i], min_T_values[i], weight_change_range[i], restart_period_values[i])) 
               for i in range(num_processes)]

    # Collect results
    solutions = [res.get() for res in results]

    # Find the best solution
    best_solution, best_cost, best_weights = min(solutions, key=lambda x: x[1])

    pool.close()
    pool.join()

    print(f"Best weights: {best_weights}")

    return best_solution

def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):
    return parallel_simulated_annealing_with_cawr(K, all_orders, all_riders, dist_mat, timelimit)
