from util2 import *


def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):

    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time).astype(int)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#

    car_rider = None
    for r in all_riders:
        if r.type == 'CAR':
            car_rider = r

    all_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        all_bundles.append(new_bundle)
        car_rider.available_number -= 1

    best_obj = sum((bundle.cost for bundle in all_bundles)) / K
    print(f'Best obj = {best_obj}')


    # Very stupid random merge algorithm
    while True:

        iter = 0
        max_merge_iter = 1000
        
        print('-------------------------- new bundling')
        while iter < max_merge_iter:
            bundle1, bundle2 = select_two_bundles(all_bundles)
            new_bundle = try_merging_bundles2(K, dist_mat, all_orders, bundle1, bundle2)

            if new_bundle is not None:
                all_bundles.remove(bundle1)
                bundle1.rider.available_number += 1
                
                all_bundles.remove(bundle2)
                bundle2.rider.available_number += 1

                all_bundles.append(new_bundle)
                new_bundle.rider.available_number -= 1

                cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
                
                # print(f'Cur obj = {cur_obj}')
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    print(f'Best obj = {best_obj}')

            else:
                iter += 1

            if time.time() - start_time > timelimit:
                break

        if time.time() - start_time > timelimit:
            break

        print('------------------------------------- change rider')

        for bundle in all_bundles:
            new_rider = get_cheaper_available_riders(all_riders, bundle.rider)
            if new_rider is not None:
                old_rider = bundle.rider
                if try_bundle_rider_changing(all_orders, bundle, new_rider):
                    old_rider.available_number += 1
                    new_rider.available_number -= 1

                if time.time() - start_time > timelimit:
                    break       

        cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
        # print(f'cur_obj = {cur_obj}')
        if cur_obj < best_obj:
            best_obj = cur_obj
            print(f'Best obj = {best_obj}')

    print(f'Last obj = {cur_obj}')

    # Solution is a list of bundle information
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    #------------- End of custom algorithm code--------------#



    return solution
    