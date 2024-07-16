from util_0702 import *


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
    timelimit=60
    init_availables = [rider.available_number for rider in all_riders]

    car_rider = [rider for rider in all_riders if rider.type == 'CAR'][0]
    bike_rider = [rider for rider in all_riders if rider.type == 'BIKE'][0]
    walk_rider = [rider for rider in all_riders if rider.type == 'WALK'][0]

    init_availables = [rider.available_number for rider in all_riders]

    min_init_cost = inf
    min_init_cost_bundle = []
    min_init_cost_rider_availables = []
    
    weight1, weight2, weight3 = 1, -1, 1
    lr = 0.4
    momentum = 0.9
    prev_update1, prev_update2, prev_update3 = 0, 0, 0
    bias = 0.04
    # 각 weight의 비율을 다르게 설정
    rate1 = 0.7
    rate2 = 1
    rate3 = 0.5
    epoch=0
    while True:
        epoch+=1
        if epoch==2:
            avg_time =time.time() - start_time 
        if time.time() - start_time > 58-avg_time:
            print(f"{epoch}time out")
            break
        bundles, result_rider_availables, cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
            K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3)
        if cost < min_init_cost:
            print(f"{epoch} break")
            min_init_cost = cost
            min_init_cost_bundles = bundles
            min_init_cost_rider_availables = result_rider_availables
            
            update1 = lr * weight1 * rate1
            update2 = lr * weight2 * rate2
            # update3 = lr * weight3 * rate3
            
            weight1 += update1 + momentum * prev_update1
            weight2 += update2 + momentum * prev_update2
            # weight3 += update3 + momentum * prev_update3
            
            prev_update1, prev_update2 = update1, update2
        else:
            update1 = lr * weight1 * rate1
            update2 = lr * weight2 * rate2
            # update3 = lr * weight3 * rate3
            
            weight1 -= update1 + momentum * prev_update1
            weight2 -= update2 + momentum * prev_update2
            # weight3 -= update3 + momentum * prev_update3
            
            prev_update1, prev_update2 = update1, update2
        weight1 += bias
        weight2 += bias
        # weight3 += bias
    print(f"time: {time.time() - start_time}")
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
    