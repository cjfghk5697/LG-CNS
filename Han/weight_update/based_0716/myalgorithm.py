from util_0716 import *


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

    init_availables = [rider.available_number for rider in all_riders]

    car_rider = [rider for rider in all_riders if rider.type == 'CAR'][0]
    bike_rider = [rider for rider in all_riders if rider.type == 'BIKE'][0]
    walk_rider = [rider for rider in all_riders if rider.type == 'WALK'][0]

    all_orders = ALL_ORDERS
    all_riders = ALL_RIDERS
    dist_mat = DIST

    start_time = time.time()

    car_rider = [rider for rider in all_riders if rider.type == 'CAR'][0]
    bike_rider = [rider for rider in all_riders if rider.type == 'BIKE'][0]
    walk_rider = [rider for rider in all_riders if rider.type == 'WALK'][0]

    init_availables = [rider.available_number for rider in all_riders]

    min_init_cost = inf
    min_init_cost_bundle = []
    min_init_cost_rider_availables = []

    weights = [1, -1, 0, -2] # r_diff weight, start_end_diff weight, d_diff weight, time window length weight

    min_cost = inf
    min_cost_init_method_num = -1
    for init_method_num in range(3):
        _, _, cost = select_init_bundle_method(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weights, init_method_num)
        
        if cost < min_cost:
            min_cost = cost
            min_cost_init_method_num = init_method_num

    _, _, n_cost = select_init_bundle_method(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weights, init_method_num + 3)

    if n_cost < cost:
        min_cost_init_method_num = init_method_num + 3

    lr = 0.7
    momentum = 0.9
    prev_update1, prev_update2, prev_update3, prev_update4 = 0, 0, 0, 0
    bias = 0.04
    # 각 weight의 비율을 다르게 설정
    rate1 = 0.5
    rate2 = 1
    rate3 = 0.5
    rate4 = 0.5
    epoch = 0
    avg_time = 0
    while True:
        epoch+=1
        if epoch==2:
            avg_time = time.time() - start_time 
        if time.time() - start_time > 58 - avg_time:
            break
        bundles, result_rider_availables, cost = select_init_bundle_method(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weights, min_cost_init_method_num)

        if cost < min_init_cost:
            min_init_cost = cost
            min_init_cost_bundles = bundles
            min_init_cost_rider_availables = result_rider_availables
            
            update1 = lr * weights[0] * rate1
            update2 = lr * weights[1] * rate2
            update3 = lr * weights[2] * rate3
            update4 = lr * weights[3] * rate4
            
            weights[0] += update1 + momentum * prev_update1
            weights[1] += update2 + momentum * prev_update2
            weights[2] += update3 + momentum * prev_update3
            weights[3] += update4 + momentum * prev_update4
            print(cost)
        else:
            update1 = lr * weights[0] * rate1
            update2 = lr * weights[1] * rate2
            update3 = lr * weights[2] * rate3
            update4 = lr * weights[3] * rate4
            
            weights[0] -= update1 + momentum * prev_update1
            weights[1] -= update2 + momentum * prev_update2
            weights[2] -= update3 + momentum * prev_update3
            weights[3] -= update4 + momentum * prev_update4
            
        prev_update1, prev_update2, prev_update3, prev_update4 = update1, update2, update3, update4

        weights[0] += bias
        weights[1] += bias
        weights[2] += bias
        weights[3] += bias

    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = min_init_cost_rider_availables[rider_i]
    all_bundles = min_init_cost_bundles

    #------------- End of custom algorithm code--------------#

    solution = [
            # rider type, shop_seq, dlv_seq
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in all_bundles
    ]

    return solution
    