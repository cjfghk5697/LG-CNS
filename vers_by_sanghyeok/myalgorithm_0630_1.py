from util_0630 import *


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

    ## Kruskal optimization

    inf = float('inf')

    init_availables = [rider.available_number for rider in all_riders]

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

    for weight1 in [0, 1]:
        for weight2 in [-3, -2, -1.5, -1, -0.5, 0, 0.5]:
            bundles, result_rider_availables, cost = get_init_bundle_4_order_bundle_maker(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2)

            if cost < min_init_cost:
                min_init_cost = cost
                min_init_cost_bundles = bundles
                min_init_cost_rider_availables = result_rider_availables

    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = min_init_cost_rider_availables[rider_i]
    all_bundles = min_init_cost_bundles


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


    ## ----------------- 커스텀 배달원 재할당 코드 -----------------------

    all_bundles, rider_availables = check_reassign_riders(K, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, all_bundles)
    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = rider_availables[rider_i]


    #------------- End of custom algorithm code--------------#

    solution = [
            # rider type, shop_seq, dlv_seq
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in all_bundles
    ]

    return solution
    