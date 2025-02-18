import math, random, sys, os
from qubots.base_problem import BaseProblem

def read_elem(filename):

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)

    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

def compute_distance_matrix(customers_x, customers_y):
    nb = len(customers_x)
    matrix = [[0 for _ in range(nb)] for _ in range(nb)]
    for i in range(nb):
        matrix[i][i] = 0
        for j in range(i+1, nb):
            d = math.sqrt((customers_x[i]-customers_x[j])**2 + (customers_y[i]-customers_y[j])**2)
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix

def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb = len(customers_x)
    dist = [0]*nb
    for i in range(nb):
        d = math.sqrt((depot_x - customers_x[i])**2 + (depot_y - customers_y[i])**2)
        dist[i] = d
    return dist

def read_input_pdptw(filename):
    file_it = iter(read_elem(filename))
    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))
    # Skip two tokens (e.g., speed, etc.) if present
    next(file_it)
    next(file_it)
    next(file_it)
    depot_x = int(next(file_it))
    depot_y = int(next(file_it))
    # Skip two tokens
    next(file_it)
    next(file_it)
    max_horizon = int(next(file_it))
    for _ in range(3):
        next(file_it)
    customers_x = []
    customers_y = []
    demands = []
    earliest = []
    latest = []
    service_time = []
    pick_up_index = []
    delivery_index = []
    i = 0
    while True:
        token = next(file_it, None)
        if token is None:
            break
        # Customer IDs are provided; we subtract 1 for 0-indexing.
        idx = int(token) - 1
        customers_x.append(int(next(file_it)))
        customers_y.append(int(next(file_it)))
        demands.append(int(next(file_it)))
        ready = int(next(file_it))
        due = int(next(file_it))
        stime = int(next(file_it))
        pick = int(next(file_it))
        delivery = int(next(file_it))
        earliest.append(ready)
        latest.append(due + stime)  # Latest end = due date + service time
        service_time.append(stime)
        pick_up_index.append(pick - 1)    # if -1 then it's a pickup
        delivery_index.append(delivery - 1)
        i += 1
    nb_customers = i
    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)
    return (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots,
            demands, service_time, earliest, latest, pick_up_index, delivery_index, max_horizon)

class PDPTWProblem(BaseProblem):
    """
    Pickup-and-Delivery Problem with Time Windows (PDPTW) for Qubots.
    
    A fleet of vehicles with uniform capacity must serve customers by picking up and then delivering items,
    subject to time window constraints. Vehicles start and end at a common depot.
    
    For each customer (node), the data include:
      - demand, service time, earliest start time, latest end time,
      - a pickup/delivery indicator: if pick_up_index[i] == -1 then customer i is a pickup,
        and its corresponding delivery is given by delivery_index[i].
    
    A candidate solution is a dictionary with key "routes" mapping to a list (of length nb_trucks)
    of routes. Each route is a list of customer indices (0-indexed) in the order they are visited by that truck.
    
    The evaluation computes:
      - The cumulative demand along each route (which must not exceed truck capacity).
      - The travel distance (from depot to first customer, between customers, and back to depot).
      - The service completion times for each visit using time windows.
      - The lateness as the extra time beyond the latest allowed finish (per visit and upon return).
      - For each pickup–delivery pair (pickup indicated by pick_up_index[i] == -1),
        the pickup and its corresponding delivery must occur in the same route with the pickup coming first.
    
    The lexicographic objectives are:
      1. Minimize total lateness (which must be 0 for a feasible solution),
      2. Minimize the number of trucks used,
      3. Minimize the total travel distance.
    These are combined here into one scalar objective using large weights.
    """
    def __init__(self, instance_file: str, **kwargs):
        (self.nb_customers, self.nb_trucks, self.truck_capacity, self.dist_matrix,
         self.dist_depot, self.demands, self.service_time, self.earliest,
         self.latest, self.pick_up_index, self.delivery_index, self.max_horizon) = read_input_pdptw(instance_file)
    
    def evaluate_solution(self, solution) -> float:
        # Expect solution to be a dictionary with key "routes" mapping to a list of routes (one per truck).
        if not isinstance(solution, dict) or "routes" not in solution:
            return 1e15
        routes = solution["routes"]
        if not isinstance(routes, list) or len(routes) != self.nb_trucks:
            return 1e15
        
        # Check that each customer is assigned exactly once.
        assigned = []
        for route in routes:
            if not isinstance(route, list):
                return 1e15
            assigned.extend(route)
        if sorted(assigned) != list(range(self.nb_customers)):
            return 1e15

        total_lateness = 0.0
        total_distance = 0.0
        trucks_used = 0
        
        # For each pickup-delivery pair, enforce that pickup occurs before delivery in the same route.
        for i in range(self.nb_customers):
            if self.pick_up_index[i] == -1:
                # i is a pickup; its corresponding delivery is delivery_index[i]
                found = False
                for route in routes:
                    if i in route and self.delivery_index[i] in route:
                        if route.index(i) < route.index(self.delivery_index[i]):
                            found = True
                            break
                if not found:
                    return 1e15

        # Evaluate each truck route.
        for route in routes:
            if len(route) == 0:
                continue
            trucks_used += 1
            # Capacity: total demand in route must not exceed truck_capacity.
            route_demand = sum(self.demands[i] for i in route)
            if route_demand > self.truck_capacity:
                return 1e15
            # Compute route travel distance: from depot to first customer, between customers, then back to depot.
            r_dist = self.dist_depot[route[0]]
            for idx in range(1, len(route)):
                r_dist += self.dist_matrix[route[idx-1]][route[idx]]
            r_dist += self.dist_depot[route[-1]]
            total_distance += r_dist
            
            # Compute service completion times and lateness.
            # Let end_time[0] = max( earliest[route[0]], dist_depot[route[0]] ) + service_time[route[0]]
            end_times = []
            s = self.dist_depot[route[0]]
            t0 = max(self.earliest[route[0]], s) + self.service_time[route[0]]
            end_times.append(t0)
            for idx in range(1, len(route)):
                travel = self.dist_matrix[route[idx-1]][route[idx]]
                s = end_times[idx-1] + travel
                t_i = max(self.earliest[route[idx]], s) + self.service_time[route[idx]]
                end_times.append(t_i)
            # Home lateness: time to return to depot
            home_time = end_times[-1] + self.dist_depot[route[-1]]
            home_lateness = max(0, home_time - self.max_horizon)
            route_lateness = home_lateness
            for idx in range(len(route)):
                route_lateness += max(0, end_times[idx] - self.latest[route[idx]])
            total_lateness += route_lateness
        
        # Lexicographic objective: first, total_lateness should be 0 for feasibility.
        # We combine objectives using weights.
        objective = total_lateness * 1e12 + trucks_used * 1e6 + total_distance
        return objective
    
    def random_solution(self):
        # Generate a random partition of customers among trucks.
        routes = [[] for _ in range(self.nb_trucks)]
        custs = list(range(self.nb_customers))
        random.shuffle(custs)
        for i, cust in enumerate(custs):
            routes[i % self.nb_trucks].append(cust)
        for route in routes:
            random.shuffle(route)
        return {"routes": routes}
