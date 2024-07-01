from util2 import *


def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):

    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time).astype(int)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#

    ## Kruskal optimization

    init_availables = [rider.available_number for rider in all_riders]

    def get_init_bundle(weight1, weight2):
        for rider_i in range(3):
            all_riders[rider_i].available_number = init_availables[rider_i]

        def find(v):
            while v != parent[v]:
                parent[v] = parent[parent[v]]
                v = parent[v]

            return v

        def union(a, b, new_bundle):
            if a > b:
                a, b = b, a

            parent[b] = a
            all_bundles[a] = new_bundle

        for r in all_riders:
            r.T = np.round(dist_mat/r.speed + r.service_time).astype(int)

        car_rider = all_riders[2]

        all_bundles = []
        for ord in all_orders:
            new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
            car_rider.available_number -= 1
            all_bundles.append(new_bundle)

        for i in range(len(all_bundles)):
            bundle = all_bundles[i]

            shop_seq = bundle.shop_seq

            xs_s_sum = 0
            ys_s_sum = 0

            xs_e_sum = 0
            ys_e_sum = 0

            readytimes_sum = 0
            deadlines_sum = 0

            shop_seq_len = len(shop_seq)

            for order_num in shop_seq:
                order = all_orders[order_num]

                xs_s_sum += order.shop_lat
                ys_s_sum += order.shop_lon

                xs_e_sum += order.dlv_lat
                ys_e_sum += order.dlv_lon

                readytimes_sum += order.ready_time
                deadlines_sum += order.deadline

            xs_s_avg = xs_s_sum / shop_seq_len
            ys_s_avg = ys_s_sum / shop_seq_len

            xs_e_avg = xs_e_sum / shop_seq_len
            ys_e_avg = ys_e_sum / shop_seq_len

            readytimes_avg = readytimes_sum / shop_seq_len
            deadlines_avg = deadlines_sum / shop_seq_len

            avg_info = [xs_s_avg, ys_s_avg, xs_e_avg, ys_e_avg, readytimes_avg, deadlines_avg]

            bundle.avg_info = avg_info

        edges = []
        for i in range(len(all_bundles)):
            for j in range(i + 1, len(all_bundles)):
                avg_info1 = all_bundles[i].avg_info
                avg_info2 = all_bundles[j].avg_info

                sx1, sy1, ex1, ey1, r1, d1 = avg_info1
                sx2, sy2, ex2, ey2, r2, d2 = avg_info2

                r_diff = abs(r1 - r2)
                d_diff = abs(d1 - d2)

                start_end_diff = get_dist_by_coords((sx1 + sx2) / 2, (sy1 + sy2) / 2, (ex1 + ex2) / 2, (ey1 + ey2) / 2)

                dist1 = dist_mat[i][j]
                dist2 = dist_mat[i + K][j + K]

                diff_score = dist1 + dist2 + r_diff * weight1 + d_diff * weight1 + start_end_diff * weight2

                edges.append((i, j, diff_score))

        parent = list(range(K))
        edges.sort(key=lambda x: x[2])

        for bundle_num1, bundle_num2, diff_score in edges:
            rbn1, rbn2 = find(bundle_num1), find(bundle_num2)

            if rbn1 == rbn2:
                continue
            
            new_bundle = try_merging_bundles4(K, dist_mat, all_orders, all_riders, all_bundles[rbn1], all_bundles[rbn2])
            if new_bundle is not None:
                all_bundles[rbn1].rider.available_number += 1
                all_bundles[rbn2].rider.available_number += 1
                new_bundle.rider.available_number -= 1

                union(rbn1, rbn2, new_bundle)

        parent = [find(v) for v in parent]

        result_bundles = [all_bundles[v] for v in set(parent)]
        result_availables = [rider.available_number for rider in all_riders]

        return result_bundles, result_availables, sum((bundle.cost for bundle in result_bundles)) / K

    # -------------------------------------------
    # all_bundles = []

    # car_rider = None
    # for r in all_riders:
    #     if r.type == 'CAR':
    #         car_rider = r

    # for ord in all_orders:
    #     new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
    #     all_bundles.append(new_bundle)
    #     car_rider.available_number -= 1
    # ------------------------------------------------

    min_init_cost = 100000000000000000
    min_init_cost_bundles = []
    min_init_cost_rider_availables = []
    for weight1 in [0, 1]:
        for weight2 in [-3, -2, -1.5, -1, -0.5, 0, 0.5]:
            bundles, result_rider_availables, cost = get_init_bundle(weight1, weight2)

            if cost < min_init_cost:
                min_init_cost = cost
                min_init_cost_bundles = bundles
                min_init_cost_rider_availables = result_rider_availables

    for rider_i in range(3):
        all_riders[rider_i].available_number = min_init_cost_rider_availables[rider_i]
    all_bundles = min_init_cost_bundles

    #------------  배달원 변경 ----------------
    for bundle in all_bundles:
        new_rider = get_cheaper_available_riders(all_riders, bundle.rider)
        if new_rider is not None:
            old_rider = bundle.rider

            check_result = check_bundle_rider_changing(all_orders, bundle, new_rider)
            if check_result:
                bundle.shop_seq = check_result[0]
                bundle.dlv_seq = check_result[1]
                bundle.rider = check_result[2]
                bundle.update_cost()

                old_rider.available_number += 1
                new_rider.available_number -= 1


    #------------- End of custom algorithm code--------------#

    solution = [
            # rider type, shop_seq, dlv_seq
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in all_bundles
    ]

    return solution
    