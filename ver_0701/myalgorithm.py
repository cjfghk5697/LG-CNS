from util_0701 import *


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

    init_availables = [rider.available_number for rider in all_riders]

    min_init_cost = inf
    min_init_cost_bundle = []
    min_init_cost_rider_availables = []

    # try_merging_bundles_by_dist // 모든 배달원과 모든 조합에서의 최소 거리를 보는 케이스
    # try_merging_bundles_by_dist_walk_prefered // 위와 동일하지만 우선적으로 도보 배달원을 할당하는 케이스
    # try_merging_bundles_by_cost // 번들 합칠 때 거리가 아닌 비용을 보는 케이스

    for weight1, weight2 in [(1, -1), (1, -2), (1, -2.5), (1, -3), (1, -3.5), (1, -4)]:
            bundles, result_rider_availables, cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2)

            if cost < min_init_cost:
                min_init_cost = cost
                min_init_cost_bundles = bundles
                min_init_cost_rider_availables = result_rider_availables

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
    