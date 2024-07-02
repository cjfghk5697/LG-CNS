from util3 import *


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

    #----------------------- merge by cos sim -----------------------#

    pq = []
    weight = []
    ch = [0 for i in range(K)]

    sz = len(all_bundles)
    for i in range(0, sz):
        for j in range(i + 1, sz):
            cur_weight = get_cos_based_weight(all_orders, dist_mat, all_bundles[i], all_bundles[j], i, j)
            heapq.heappush(pq, (cur_weight, [i, j]))

    is_merged = True
    iter_cnt = 0

    while is_merged:
        if time.time() - start_time > timelimit - 1: 
                    break
        is_merged = False
        iter_cnt += 1

        while pq:
            cur_weight, [idx1, idx2] = heapq.heappop(pq)
            if ch[idx1] == 1 or ch[idx2] == 1:
                continue

            bundle1 = all_bundles[idx1]
            bundle2 = all_bundles[idx2]
            new_bundle = try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2)
            if new_bundle is not None:
                ch[idx1], ch[idx2] = 1, 1
                new_bundles.append(new_bundle)
                is_merged = True

        if is_merged:
            for i in range(K):
                if ch[i] == 0:
                    new_bundles.append(all_bundles[i])

            for cur_bundle in new_bundles:
                make_it_optimal(all_orders, all_riders, cur_bundle)

            all_bundles = deepcopy(new_bundles)
            del new_bundles[:]
    
    print("iter_cnt :",iter_cnt)
    #----------------------- merge by cos sim -----------------------#

    # Solution is a list of bundle information
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    #------------- End of custom algorithm code--------------#

    return solution
    