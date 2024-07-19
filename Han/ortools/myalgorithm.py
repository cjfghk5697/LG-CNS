from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from util import *

def solve_tsp(distance_matrix):
    tsp_size = len(distance_matrix)
    num_routes = 1
    depot = 0

    if tsp_size > 0:
        routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        def distance_callback(from_index, to_index):
            return distance_matrix[from_index][to_index]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            index = routing.Start(0)
            plan_output = []
            while not routing.IsEnd(index):
                plan_output.append(routing.IndexToNode(index))
                index = assignment.Value(routing.NextVar(index))
            plan_output.append(routing.IndexToNode(index))
            return plan_output
        else:
            return None
    else:
        return None

def create_distance_matrix(K, dist_mat, shop_seq, dlv_seq):
    indices = shop_seq + [k + K for k in dlv_seq]
    size = len(indices)
    distance_matrix = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            distance_matrix[i][j] = dist_mat[indices[i], indices[j]]
    return distance_matrix, indices

def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):

    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    # Initialize bundles with individual orders
    all_bundles = []
    car_rider = next(r for r in all_riders if r.type == 'CAR')
    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        all_bundles.append(new_bundle)
        car_rider.available_number -= 1

    best_obj = sum((bundle.cost for bundle in all_bundles)) / K
    print(f'Best obj = {best_obj}')

    # Main algorithm loop
    while time.time() - start_time < timelimit:
        bundle1, bundle2 = select_two_bundles(all_bundles)
        new_bundle = try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2)

        if new_bundle is not None:
            all_bundles.remove(bundle1)
            bundle1.rider.available_number += 1

            all_bundles.remove(bundle2)
            bundle2.rider.available_number += 1

            all_bundles.append(new_bundle)
            new_bundle.rider.available_number -= 1

            cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
            if cur_obj < best_obj:
                best_obj = cur_obj
                print(f'Best obj = {best_obj}')

        for bundle in all_bundles:
            new_rider = get_cheaper_available_riders(all_riders, bundle.rider)
            if new_rider is not None:
                old_rider = bundle.rider
                if try_bundle_rider_changing(all_orders, dist_mat, bundle, new_rider):
                    old_rider.available_number += 1
                    new_rider.available_number -= 1

                cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    print(f'Best obj = {best_obj}')

    # Optimize each bundle using OR-Tools
    for bundle in all_bundles:
        distance_matrix, indices = create_distance_matrix(K, dist_mat, bundle.shop_seq, bundle.dlv_seq)
        tsp_solution = solve_tsp(distance_matrix)
        if tsp_solution:
            tsp_solution_shop = [indices[i] for i in tsp_solution if i < len(bundle.shop_seq)]
            tsp_solution_dlv = [indices[i] - K for i in tsp_solution if i >= len(bundle.shop_seq)]
            bundle.shop_seq = tsp_solution_shop
            bundle.dlv_seq = tsp_solution_dlv
            bundle.total_dist = get_total_distance(K, dist_mat, bundle.shop_seq, bundle.dlv_seq)
            bundle.update_cost()

    # Create final solution
    solution = [
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    return solution
