# 8-4_bundling_opt_ver1_KruskalDiverseRandom_KCasework
# 크루스칼에 랜덤 기존 가중치 및 추가 가중치 부여, K >= 200 케이스워크, SA 포함

from util_0804_2 import *

def simulated_annealing(K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, is_allow_worse_case, init_availables, order_comb_possibility):
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
            car_rider, K, cur_solution, all_riders, all_orders, dist_mat, T, is_pre_decreased, init_availables, order_comb_possibility)
        
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

    all_bundles = []
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
    min_init_cost_bundles = []
    min_init_cost_rider_availables = []

    weight_grid = []

    if K >= 200:
        weight1 = 0
        for _ in range(5):
            weight2 = 0
            for _ in range(7):
                weight_grid.append((weight1, weight2, 0))
                weight2 -= 0.5
            weight1 += 0.5
    else:
        weight1 = 0
        for _ in range(6):
            weight2 = 0
            for _ in range(9):
                for weight3 in range(-1, 3):
                    weight_grid.append((weight1, weight2, weight3))
                weight2 -= 0.5
            weight1 += 0.5  

    weight_grid.sort(key=lambda x: (x[0] + x[1] + x[2]))

    min_weight_comb = []

    for weight1, weight2, weight3 in weight_grid:
        cur_time = time.time()
        if cur_time - start_time > 30:
            break

        bundles, result_rider_availables, cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
                K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3, order_comb_possibility, optimized_order_perms, False)

        if cost < min_init_cost:
            min_init_cost = cost
            min_init_cost_bundles = bundles
            min_init_cost_rider_availables = result_rider_availables
            min_weight_comb = [weight1, weight2, weight3]
        
        if K <= 100:
            bundles, result_rider_availables, _ = get_init_bundle(K, all_riders, all_orders, DIST, init_availables, weight1, weight2, weight3, try_merging_bundles_by_dist_possibles_only, order_comb_possibility, optimized_order_perms, False, 4)
            for rider_i in range(3):
                ALL_RIDERS[rider_i].available_number = result_rider_availables[rider_i]

            bundles, result_rider_availables = reassign_riders(K, ALL_ORDERS, ALL_RIDERS, DIST, init_availables, bundles)
            cost = sum((bundle.cost for bundle in bundles)) / K

            if cost < min_init_cost:
                min_init_cost = cost
                min_init_cost_bundles = bundles
                min_init_cost_rider_availables = result_rider_availables
                min_weight_comb = [weight1, weight2, weight3]

    weight1, weight2, weight3 = min_weight_comb

    # print('start', min_init_cost)

    while True:
        cur_time = time.time()
        if cur_time - start_time > 30:
            break

        bundles, result_rider_availables, cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
            K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1, weight2, weight3, order_comb_possibility, optimized_order_perms, True)

        if cost < min_init_cost:
            min_init_cost = cost
            min_init_cost_bundles = bundles
            min_init_cost_rider_availables = result_rider_availables

            # print('changed', cost)

    for rider_i in range(3):
        ALL_RIDERS[rider_i].available_number = min_init_cost_rider_availables[rider_i]
    all_bundles = min_init_cost_bundles

    freeze_support()

    num_core = 4
    with Pool(num_core) as pool:
        result = pool.starmap(simulated_annealing, [[K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, 1, init_availables, order_comb_possibility],
                                                    [K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, 1, init_availables, order_comb_possibility],
                                                    [K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, 1, init_availables, order_comb_possibility],
                                                    [K, all_orders, all_riders, dist_mat, timelimit, all_bundles, start_time, car_rider, ALL_ORDERS, ALL_RIDERS, DIST, -1, init_availables, order_comb_possibility],])
        
    cost_result = [sum(bundle.cost for bundle in cur_solution) / K for cur_solution in result]
    final_solution_idx = np.argmin(cost_result)
    final_solution = result[final_solution_idx]
    # #------------- End of custom algorithm code--------------#

    solution = [
            # rider type, shop_seq, dlv_seq
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in final_solution
    ]

    return solution, min(cost_result)