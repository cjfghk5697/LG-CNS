from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from util import Order, Rider, Bundle, get_total_distance, get_total_volume, solution_check
import numpy as np
import json

def create_data_model(all_orders, all_riders, dist_mat):
    data = {}
    data['distance_matrix'] = dist_mat.tolist()  # 거리 행렬
    data['num_vehicles'] = sum(rider.available_number for rider in all_riders)  # 차량(배달원)의 총 수
    data['depot'] = 0  # 출발 지점
    data['orders'] = all_orders  # 모든 주문 정보
    data['riders'] = all_riders  # 모든 배달원 정보
    data['vehicle_capacities'] = [rider.capa for rider in all_riders for _ in range(rider.available_number)]  # 각 차량의 용량
    data['vehicle_types'] = [i for i, rider in enumerate(all_riders) for _ in range(rider.available_number)]  # 차량 유형
    data['vehicle_speeds'] = [rider.speed for rider in all_riders for _ in range(rider.available_number)]  # 차량 속도
    data['vehicle_service_times'] = [rider.service_time for rider in all_riders for _ in range(rider.available_number)]  # 서비스 시간
    return data

def create_solution(manager, routing, solution, all_orders, all_riders, dist_mat, data):
    all_bundles = []
    for vehicle_id in range(routing.vehicles()):  # 모든 차량에 대해 루프 실행
        index = routing.Start(vehicle_id)  # 차량의 시작 인덱스
        shop_seq = []  # 픽업 순서
        dlv_seq = []  # 배달 순서
        while not routing.IsEnd(index):  # 끝에 도달할 때까지 반복
            node_index = manager.IndexToNode(index)
            if node_index < len(all_orders):  # 픽업 노드
                shop_seq.append(node_index)
            else:  # 배달 노드
                dlv_seq.append(node_index - len(all_orders))
            index = solution.Value(routing.NextVar(index))
        
        if shop_seq and dlv_seq:  # 유효한 픽업 및 배달 순서가 있는 경우
            rider = all_riders[data['vehicle_types'][vehicle_id]]  # 차량(배달원) 정보 가져오기
            total_volume = get_total_volume(all_orders, shop_seq)  # 총 부피 계산
            total_dist = get_total_distance(len(all_orders), dist_mat, shop_seq, dlv_seq)  # 총 거리 계산
            new_bundle = Bundle(all_orders, rider, shop_seq, dlv_seq, total_volume, total_dist)  # 새로운 번들 생성
            all_bundles.append(new_bundle)  # 번들 리스트에 추가
    
    solution_list = [
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]
    
    total_cost = sum(bundle.cost for bundle in all_bundles)  # 총 비용 계산
    avg_cost = total_cost / len(all_orders)  # 평균 비용 계산
    
    return solution_list, avg_cost

def algorithm(all_orders, all_riders, dist_mat, timelimit=60):
    data = create_data_model(all_orders, all_riders, dist_mat)

    # RoutingIndexManager는 각 노드의 인덱스를 관리합니다.
    # 여기서는 모든 주문 및 배달 지점과 차량(배달원) 정보를 관리합니다.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # RoutingModel은 라우팅 문제를 정의하고 해결하는 모델입니다.
    # 이는 차량 경로 문제(VRP)를 설정하고 해결하는 데 사용됩니다.
    routing = pywrapcp.RoutingModel(manager)

    # 거리 콜백 함수: 두 노드 간의 거리를 반환합니다.
    # 이 함수는 각 노드 간의 이동 비용(거리)을 계산하는 데 사용됩니다.
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    # 거리 콜백 함수를 등록하고, 모든 차량에 대해 비용 평가자를 설정합니다.
    # RegisterTransitCallback을 사용하여 이동 비용을 계산하는 콜백 함수를 등록합니다.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 수요 콜백 함수: 각 노드의 수요(여기서는 주문의 부피)를 반환합니다.
    # 이 함수는 각 노드의 용량 제한을 확인하는 데 사용됩니다.
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['orders'][from_node].volume

    # 수요 콜백 함수를 등록하고, 차량 용량을 고려한 차원을 추가합니다.
    # AddDimensionWithVehicleCapacity를 사용하여 차량의 용량 제한을 설정합니다.
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # 허용 오차
        data['vehicle_capacities'],  # 차량의 용량
        True,  # 이 용량을 강제로 적용할지 여부
        'Capacity')  # 차원 이름

    # 시간 콜백 함수: 두 노드 간의 이동 시간(거리/속도 + 서비스 시간)을 반환합니다.
    # 이 함수는 각 차량 유형별로 이동 시간을 계산하는 데 사용됩니다.
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        vehicle_type = data['vehicle_types'][routing.VehicleVar(from_index).Value()]
        speed = data['vehicle_speeds'][vehicle_type]
        service_time = data['vehicle_service_times'][vehicle_type]
        return data['distance_matrix'][from_node][to_node] / speed + service_time

    # 시간 콜백 함수를 등록하고, 시간 차원을 추가합니다.
    # AddDimension을 사용하여 시간 제약 조건을 설정합니다.
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        30,  # 허용 오차
        10000,  # 최대 시간
        False,  # 이 시간을 강제로 적용할지 여부
        'Time')  # 차원 이름

    # 시간 차원: 각 노드의 시간 윈도우를 설정합니다.
    # 각 주문의 준비 시간과 배달 마감 시간을 설정합니다.
    time_dimension = routing.GetDimensionOrDie('Time')
    for order in data['orders']:
        index = manager.NodeToIndex(order.id)
        ready_time = order.ready_time if order.ready_time >= 0 else 0
        deadline = order.deadline if order.deadline > ready_time else ready_time + 10000
        time_dimension.CumulVar(index).SetRange(ready_time, deadline)

    # 검색 매개변수: 초기 솔루션 전략과 메타 휴리스틱을 설정합니다.
    # 경로 탐색을 위한 매개변수를 설정합니다.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = timelimit

    # 라우팅 문제를 해결합니다.
    # 설정된 매개변수를 사용하여 경로 문제를 해결합니다.
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        print("No solution found!")
        return None, None

    return create_solution(manager, routing, solution, data['orders'], data['riders'], dist_mat, data)

# 테스트 실행
problem_file = f"C:/Users/cjfgh/vscode/LG-CNS/@shared/testfile/STAGE1_1.json"

with open(problem_file, 'r') as f:
    prob = json.load(f)

K = prob['K']
ALL_ORDERS = [Order(order_info) for order_info in prob['ORDERS']]
ALL_RIDERS = [Rider(rider_info) for rider_info in prob['RIDERS']]
DIST = np.array(prob['DIST'])

solution, avg_cost = algorithm(ALL_ORDERS, ALL_RIDERS, DIST)

if solution is None or not isinstance(solution, list):
    print("Solution is not valid or an exception occurred.")
else:
    print("Solution found:")
    print(solution)
    print(f"Average cost: {avg_cost}")
