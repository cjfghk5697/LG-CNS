from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from util import Order, Rider, Bundle, get_total_distance, get_total_volume, solution_check
import numpy as np
import json

def create_data_model(all_orders, all_riders, dist_mat):
    data = {}
    data['distance_matrix'] = dist_mat.tolist()
    data['num_vehicles'] = sum(rider.available_number for rider in all_riders)
    data['depot'] = 0
    data['orders'] = all_orders
    data['riders'] = all_riders
    data['vehicle_capacities'] = [rider.capa for rider in all_riders for _ in range(rider.available_number)]
    data['vehicle_types'] = [i for i, rider in enumerate(all_riders) for _ in range(rider.available_number)]
    data['vehicle_speeds'] = [rider.speed for rider in all_riders for _ in range(rider.available_number)]
    data['vehicle_service_times'] = [rider.service_time for rider in all_riders for _ in range(rider.available_number)]
    data['fixed_costs'] = [rider.fixed_cost for rider in all_riders for _ in range(rider.available_number)]
    data['variable_costs'] = [rider.var_cost for rider in all_riders for _ in range(rider.available_number)]
    return data

def create_solution(manager, routing, solution, all_orders, all_riders, dist_mat, data):
    all_bundles = []
    total_cost = 0
    total_dist = 0
    for vehicle_id in range(routing.vehicles()):
        index = routing.Start(vehicle_id)
        shop_seq = []
        dlv_seq = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index < len(all_orders):
                shop_seq.append(node_index)
            else:
                dlv_seq.append(node_index - len(all_orders))
            index = solution.Value(routing.NextVar(index))
        
        if shop_seq and dlv_seq:
            rider = all_riders[data['vehicle_types'][vehicle_id]]
            total_volume = get_total_volume(all_orders, shop_seq)
            dist = get_total_distance(len(all_orders), dist_mat, shop_seq, dlv_seq)
            new_bundle = Bundle(all_orders, rider, shop_seq, dlv_seq, total_volume, dist)
            all_bundles.append(new_bundle)
            total_cost += rider.calculate_cost(dist)
            total_dist += dist

    solution_list = [
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]
    
    avg_cost = total_cost / len(all_orders)
    
    return solution_list, avg_cost, total_dist

def algorithm(all_orders, all_riders, dist_mat, timelimit=60):
    rider_priority = ['BIKE', 'CAR', 'WALK']
    all_riders = sorted(all_riders, key=lambda r: rider_priority.index(r.type))

    data = create_data_model(all_orders, all_riders, dist_mat)

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['orders'][from_node].volume

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        data['vehicle_capacities'],
        True,
        'Capacity')

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        vehicle_type = data['vehicle_types'][routing.VehicleVar(from_index).Value()]
        speed = data['vehicle_speeds'][vehicle_type]
        service_time = data['vehicle_service_times'][vehicle_type]
        return data['distance_matrix'][from_node][to_node] / speed + service_time

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        30,
        10000,
        False,
        'Time')

    time_dimension = routing.GetDimensionOrDie('Time')
    for order in data['orders']:
        index = manager.NodeToIndex(order.id)
        ready_time = order.ready_time if order.ready_time >= 0 else 0
        deadline = order.deadline if order.deadline > ready_time else ready_time + 10000
        time_dimension.CumulVar(index).SetRange(ready_time, deadline)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = timelimit

    while True:
        solution = routing.SolveWithParameters(search_parameters)
        if not solution:
            print("No solution found!")
            return None, None

        sol_list, avg_cost, total_dist = create_solution(manager, routing, solution, data['orders'], data['riders'], dist_mat, data)
        if solution_check(len(data['orders']), data['orders'], data['riders'], dist_mat, sol_list) == (total_cost, total_dist):
            return sol_list, avg_cost
        routing.CloseModelWithParameters(search_parameters)

problem_file = f"C:/Users/cjfgh/vscode/LG-CNS/@shared/testfile/STAGE1_1.json"

with open(problem_file, 'r') as f:
    prob = json.load(f)

K = prob['K']
ALL_ORDERS = [Order(order_info) for order_info in prob['ORDERS']]
ALL_RIDERS = [Rider(rider_info) for rider_info in prob['RIDERS']]
DIST = np.array(prob['DIST'])
prob['DIST'].insert(0,[0 for _ in range(len(prob['DIST'][0]))])
for i in range(len(prob['DIST'])):
    prob['DIST'][i].insert(0,0)
print(prob['DIST'][1])
solution, avg_cost = algorithm(ALL_ORDERS, ALL_RIDERS, DIST)

if solution is None or not isinstance(solution, list):
    print("Solution is not valid or an exception occurred.")
else:
    print("Solution found:")
    print(solution)
    print(f"Average cost: {avg_cost}")
