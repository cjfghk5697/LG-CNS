import time

from util3 import *

def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat / r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    # ------------- Custom algorithm code starts from here --------------#

    car_rider = all_riders[2]
    all_bundles = []
    org_all_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        org_all_bundles.append(new_bundle)
        car_rider.available_number -= 1

    # ------------- SA init ------------- #
    cur_temp = 100000
    critical_temp = 0.005
    delta = 0.9999
    k_ = 1
    pre_proba = 0.5
    cur_optimal_cost = sum((bundle.cost for bundle in org_all_bundles)) / K
    cur_optimal_bundles = org_all_bundles

    print(all_riders[0].available_number)
    print(all_riders[1].available_number)
    print(all_riders[2].available_number)
    print("##########################")

    while len(org_all_bundles) >= 3:
        if time.time() - start_time > timelimit / 2:
            break
        bundle1, bundle2, bundle3 = select_three_bundles(org_all_bundles)

        new_bundle1, tmp_rider = try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2)
        if new_bundle1 is None:
            continue

        new_bundle2, tmp_rider = try_merging_bundles(K, dist_mat, all_orders, new_bundle1, bundle3)
        if new_bundle2 is None:
            continue

        org_all_bundles.remove(bundle1)
        org_all_bundles.remove(bundle2)
        org_all_bundles.remove(bundle3)
        all_bundles.append(new_bundle2)
        new_bundle2.rider.available_number += 2

    print(all_bundles)
    print(all_riders[0].available_number)
    print(all_riders[1].available_number)
    print(all_riders[2].available_number)
    print("##########################")

    while org_all_bundles:
        cur_bundle = org_all_bundles[0]
        org_all_bundles.remove(cur_bundle)
        all_bundles.append(cur_bundle)
    

    for i in range(len(all_bundles)):
        nxt_bundle = get_optimal_path(all_orders, all_bundles[i], all_bundles[i].rider)
        if nxt_bundle is not None:
            all_bundles[i] = nxt_bundle

    '''
    for cur_bundle in all_bundles:
        cost, nxt_bundle = get_optimal_path(all_orders, cur_bundle, cur_bundle.rider)
        cur_bundle = nxt_bundle
    '''
    print(all_riders[0].available_number)
    print(all_riders[1].available_number)
    print(all_riders[2].available_number)
    print("##########################")

    while cur_temp > critical_temp:
        if time.time() - start_time > timelimit - 2:
            break

        E1 = sum((bundle.cost for bundle in all_bundles)) / K

        #-------------get E2--------------#
        ret_bundles = all_bundles.copy()
        cur_cost = 0

        # 배달원 변경
        if 0.5 < random.random():
            cur_bundle = random.choice(all_bundles)
            new_rider = get_cheaper_available_riders(all_riders, cur_bundle.rider)
            if new_rider is not None and new_rider.available_number > 0:
                old_rider = cur_bundle.rider
                if try_bundle_rider_changing(all_orders, cur_bundle, new_rider):
                    old_rider.available_number += 1
                    new_rider.available_number -= 1

        # 배달 묶기
        if 0.5 < random.random():
            bundle1, bundle2 = random.sample(ret_bundles, 2)
            new_bundle, new_rider = try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2)
            if new_bundle is not None and new_rider.available_number > 0:
                ret_bundles.remove(bundle1)
                bundle1.rider.available_number += 1

                ret_bundles.remove(bundle2)
                bundle2.rider.available_number += 1

                ret_bundles.append(new_bundle)
                new_bundle.rider.available_number -= 1

        cur_cost = sum((bundle.cost for bundle in ret_bundles)) / K
        # -------------get E2--------------#

        E2 = cur_cost
        neighbor_state_bundles = ret_bundles

        p = get_cur_proba(E1, E2, k_, cur_temp)

        if p < random.random():
            all_bundles = neighbor_state_bundles
            E1 = E2

        if E1 < cur_optimal_cost:
            cur_optimal_cost = E1
            cur_optimal_bundles = all_bundles

        cur_temp *= delta

        pre_proba = p
        print(all_riders[0].available_number)
        print(all_riders[1].available_number)
        print(all_riders[2].available_number)
        print("##########################")

    solution = [
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in cur_optimal_bundles
    ]

    print(all_riders[0].available_number)
    print(all_riders[1].available_number)
    print(all_riders[2].available_number)
    # ------------- End of custom algorithm code--------------#

    return solution
