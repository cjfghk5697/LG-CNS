from han_SA_util import *

def simulated_annealing(K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, is_allow_worse_case):
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
            car_rider, K, cur_solution, all_riders, all_orders, dist_mat, T, is_pre_decreased)
        
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
        #T *= delta
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

    # weight1, weight2 = 1, -1
    # bundles, result_rider_availables, cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
    #              K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2)
    
    # if cost < min_init_cost:
    #             min_init_cost = cost
    #             min_init_cost_bundles = bundles
    #             min_init_cost_rider_availables = result_rider_availables
    
    for weight1, weight2 in [(1, -1), (1, -2), (1, -2.5), (1, -3), (1, -3.5), (1, -4)]:
        if time.time() - start_time > 20: 
            break
        bundles, result_rider_availables, cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
                 K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2)

        if cost < min_init_cost:
            min_init_cost = cost
            min_init_cost_bundles = bundles
            min_init_cost_rider_availables = result_rider_availables

    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = min_init_cost_rider_availables[rider_i]
    all_bundles = min_init_cost_bundles

    freeze_support()

    num_core = 4
    with Pool(num_core) as pool:
        result = pool.starmap(simulated_annealing, [[K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, 1],
                                                    [K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, 1],
                                                    [K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, 1],
                                                    [K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, -1],])
        
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
