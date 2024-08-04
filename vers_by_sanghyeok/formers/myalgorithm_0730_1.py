# from util import *
from util_0730 import *


def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):
    ALL_ORDERS = all_orders
    ALL_RIDERS = all_riders
    DIST = dist_mat

    start_time = time.time()

    for rider in all_riders:
        rider.T = np.round(dist_mat / rider.speed + rider.service_time).astype(int)

    solution = []

    inf = float('inf')
    timelimit = 60
    init_availables = [rider.available_number for rider in all_riders]

    min_init_cost = inf
    min_init_cost_bundle = []
    min_init_cost_rider_availables = []

    weight1, weight2, weight3 = 1, -1, 1
    lr = 0.7
    momentum = 0.9
    prev_update1, prev_update2 = 0, 0
    bias = 0.04
    rate1 = 1
    rate2 = 0.7
    epoch = 0
    avg_time = 0

    min_init_cost_bundle, min_init_cost_rider_availables, min_init_cost = get_clustered_bundle_4_order_prefered(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3)

    # while True:
    #     epoch += 1
    #     if epoch == 2:
    #         avg_time = time.time() - start_time
    #     if time.time() - start_time > 10 - avg_time:
    #         break
    #     bundles, result_rider_availables, cost = get_clustered_bundle_4_order_prefered(
    #         K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3)
    #     if cost < min_init_cost:
    #         min_init_cost = cost
    #         min_init_cost_bundle = bundles
    #         min_init_cost_rider_availables = result_rider_availables

    #         update1 = lr * weight1 * rate1
    #         update2 = lr * weight2 * rate2

    #         weight1 += update1 + momentum * prev_update1
    #         weight2 += update2 + momentum * prev_update2

    #         prev_update1, prev_update2 = update1, update2
    #         print(cost)
    #     else:
    #         update1 = lr * weight1 * rate1
    #         update2 = lr * weight2 * rate2

    #         weight1 -= update1 + momentum * prev_update1
    #         weight2 -= update2 + momentum * prev_update2

    #         prev_update1, prev_update2 = update1, update2
    #     weight1 += bias
    #     weight2 += bias

    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = min_init_cost_rider_availables[rider_i]
    all_bundles = min_init_cost_bundle

    solution = [
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    return solution
