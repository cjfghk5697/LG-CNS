from CNS_FA_util import *


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

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        all_bundles.append(new_bundle)
        car_rider.available_number -= 1

    #----------------- FA init -----------------#

    firefiy_num = 4
    _lambda_ = 0.8
    insertion_cnt_discount_factor = 0.8
    mutation_cnt_discount_factor = 0.8
    fireflies = []

    rider_cnt = []

    for _ in range(firefiy_num):
        cur_firefly = Firefly(all_bundles)
        new_firefly = deepcopy(cur_firefly)
        fireflies.append(deepcopy(new_firefly))
        random.shuffle(all_bundles)

        rider_cnt.append({'BIKE' : int(bike_rider.available_number), 
                          'WALK' : int(walk_rider.available_number),
                          'CAR' : int(car_rider.available_number)})

    #----------------- FA init -----------------#

    #----------------- FA iter -----------------#

    iter_cnt = 0
    avg_time = 0
    iter_start_time = time.time()
    
    I = [0] * firefiy_num
    while True:
        #-- for debugging --#
        # if iter_cnt > 1:
        #     break
        #-- for debugging --#

        if time.time() - start_time > timelimit - avg_time:
            break
        iter_cnt += 1

        for i in range(firefiy_num):
            I[i] = -1 * sum((bundle.cost for bundle in fireflies[i].bundles)) / K 

        for i in range(firefiy_num):
            for j in range(i + 1, firefiy_num):
                r_ij = get_hemming_dist(fireflies[i], fireflies[j])
                cnt = random.randint(2, int(r_ij * math.pow(_lambda_, 9)))

                if I[i] > I[j]:
                    insertion(fireflies[j], max(1, int(cnt / insertion_cnt_discount_factor)), rider_cnt[j], all_orders, dist_mat, K)
                    #mutation(fireflies[j], max(1, int(cnt / mutation_cnt_discount_factor)), rider_cnt[j], all_orders, all_riders)
                elif I[i] <= I[j]:
                    insertion(fireflies[i], max(1, int(cnt / insertion_cnt_discount_factor)), rider_cnt[i], all_orders, dist_mat, K)
                    #mutation(fireflies[j], max(1, int(cnt / mutation_cnt_discount_factor)), rider_cnt[j], all_orders, all_riders)
            mutation(fireflies[i], max(1, int(cnt / mutation_cnt_discount_factor)), rider_cnt[i], all_orders, all_riders)
        avg_time = (time.time() - iter_start_time) / iter_cnt

    #----------------- FA iter -----------------#

    print("iter cnt :", iter_cnt)
    print("avg time per iter :", avg_time)
    
    for i in range(firefiy_num):
            I[i] = -1 * sum((bundle.cost for bundle in fireflies[i].bundles)) / K 
    best_idx = np.argmax(I)
    print(rider_cnt[best_idx])
    all_bundles = fireflies[best_idx].bundles

    # Solution is a list of bundle information
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    #------------- End of custom algorithm code--------------#

    return solution
    
