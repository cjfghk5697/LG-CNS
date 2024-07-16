from util_0702 import *
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time

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

    min_init_cost = inf
    min_init_cost_bundle = []
    min_init_cost_rider_availables = []

    # Initialize weights
    weight1 = torch.tensor([5.0], requires_grad=True)
    weight2 = torch.tensor([5.0], requires_grad=True)
    weight3 = torch.tensor([5.0], requires_grad=True)

    # Optimizer and scheduler
    optimizer = Adam([weight1, weight2, weight3], lr=1e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)

    for epoch in range(10):
        # Zero the gradients
        optimizer.zero_grad()

        # Perform your custom algorithm step
        bundles, result_rider_availables, cost = get_init_bundle_4_order_bundle_prefered_with_reassigning_riders(
            K, ALL_RIDERS, ALL_ORDERS, DIST, init_availables, weight1.item(), weight2.item(), weight3.item()
        )

        # If cost improves, update minimum cost and weights
        if cost < min_init_cost:
            print("break")
            min_init_cost = cost
            min_init_cost_bundles = bundles
            min_init_cost_rider_availables = result_rider_availables
        
        # Perform a backward pass to compute gradients
        cost.backward()

        # Update weights
        optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Print the current weights and learning rates for debugging
        print(f"Epoch {epoch+1}: weight1 = {weight1.item()}, weight2 = {weight2.item()}, weight3 = {weight3.item()}")
        print(f"Epoch {epoch+1}: lr = {scheduler.get_last_lr()}")

    #------------- End of custom algorithm code--------------#

    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    return solution
