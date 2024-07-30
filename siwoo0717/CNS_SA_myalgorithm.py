from CNS_SA_util import *


def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):

    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#
    
    bike_rider = None
    walk_rider = None
    car_rider = None
    for r in all_riders:
        if r.type == 'BIKE':
            bike_rider = r
        elif r.type == 'WALK':
            walk_rider = r
        elif r.type == 'CAR':
            car_rider = r

    all_bundles = []
    new_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        all_bundles.append(new_bundle)
        car_rider.available_number -= 1
        
    #------------------------------------------ merge cos sim ------------------------------------------#
    
    pq = []
    ch = [0 for i in range(K)]

    sz = len(all_bundles)
    for i in range(0, sz):
        for j in range(i + 1, sz):
            cur_weight = get_cos_based_weight(all_orders, all_bundles[i], all_bundles[j])
            heapq.heappush(pq, (cur_weight, [i, j]))

    is_merged = True
    merge_iter_cnt = 0
    pre_pq_sz = len(pq)

    while is_merged:
        # if time.time() - start_time > timelimit - 1: break
        is_merged = False
        
        if time.time() - start_time > 5:
            break
        merge_iter_cnt += 1
        
        while pq:
            cur_weight, [idx1, idx2] = heapq.heappop(pq)
            if ch[idx1] == 1 or ch[idx2] == 1:
                continue

            bundle1 = all_bundles[idx1]
            bundle2 = all_bundles[idx2]
            rider1 = bundle1.rider
            rider2 = bundle2.rider
            new_bundle = try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2)
            
            if new_bundle is not None:
                new_rider = new_bundle.rider
                
                rider1.available_number += 1
                rider2.available_number += 1
                new_rider.available_number -= 1
                
                ch[idx1], ch[idx2] = 1, 1
                new_bundles.append(new_bundle)
                is_merged = True

        if is_merged:
            for i in range(len(all_bundles)):
                if ch[i] == 0:
                    new_bundles.append(all_bundles[i])

            all_bundles = deepcopy(new_bundles)
            new_bundles = []

            ch = [0 for i in range(K)]

            sz = len(all_bundles)
            for i in range(0, sz):
                if len(all_bundles[i].shop_seq) > 3: continue
                for j in range(i + 1, sz):
                    if len(all_bundles[j].shop_seq) > 3: continue
                    cur_weight = get_cos_based_weight(all_orders, all_bundles[i], all_bundles[j])
                    heapq.heappush(pq, (cur_weight, [i, j]))

        if len(pq) / pre_pq_sz > 0.9: break
        pre_pq_sz = len(pq)
            
        if merge_iter_cnt > 2 :
            break
    
    print("merge iter cnt :", merge_iter_cnt)
    print("time taken to merge:", time.time() - start_time)
    
    #------------------------------------------ merge cos sim ------------------------------------------#
    
#--------------------------------------------- SA init ---------------------------------------------#
    print(sum(bundle.cost for bundle in all_bundles) / K )
    
    hist = []
    T = 100000000
    delta = 0.995
    T_final = 0.0001
    cur_solution = deepcopy(all_bundles)
    cur_cost = sum(bundle.cost for bundle in all_bundles) / K 
    SA_iter_cnt = 0
    T_mulitiplier = 0.0000001

    # for bundle in all_bundles:
    #     if len(bundle.shop_seq) > 3:
    #         continue
    #     make_path_optimal(bundle, rider_cnt, all_orders, all_riders) 
    
    #--------------------------------------------- SA init ---------------------------------------------#

    #--------------------------------------------- SA iter ---------------------------------------------#
    
    while True:
        if time.time() - start_time > timelimit - 1 or T <= T_final: 
            break    
        SA_iter_cnt += 1
        
        new_solution, new_cost = make_new_solution(
            car_rider, K, cur_solution, all_riders, all_orders, dist_mat, T)
        
        if new_cost < cur_cost:
            cur_solution = new_solution
            cur_cost = new_cost
        elif new_cost == cur_cost:
            continue
        elif new_cost > cur_cost:
            p_discount_factor = SA_iter_cnt / 2
            p = math.exp(-(new_cost - cur_cost) / (T * T_mulitiplier / p_discount_factor))

            print(int(new_cost), int(cur_cost), int(T), 
                  -(new_cost - cur_cost) / (T * T_mulitiplier / p_discount_factor), p)
            if p > random.random():
                print("changed")
                cur_solution = new_solution
                cur_cost = new_cost
        
        T *= delta
        hist.append(cur_cost)

    print("SA iter cnt :", SA_iter_cnt)

    for bundle in cur_solution:
        # if len(bundle.shop_seq) > 3:
        #     continue
        make_path_optimal(K, dist_mat, bundle, all_orders, all_riders)  

    print(sum(bundle.cost for bundle in cur_solution) / K )
    
    plt.plot(hist)
    #--------------------------------------------- SA iter ---------------------------------------------#


    #------------- End of custom algorithm code--------------#

    solution = [
            # rider type, shop_seq, dlv_seq
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in cur_solution
    ]

    return solution
    
