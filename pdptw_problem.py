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
    matrix = [[0.0 for _ in range(nb)] for _ in range(nb)]
    for i in range(nb):
        matrix[i][i] = 0.0
        for j in range(i+1, nb):
            d = math.sqrt((customers_x[i]-customers_x[j])**2 + (customers_y[i]-customers_y[j])**2)
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix

def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb = len(customers_x)
    dist = [0.0]*nb
    for i in range(nb):
        d = math.sqrt((depot_x - customers_x[i])**2 + (depot_y - customers_y[i])**2)
        dist[i] = d
    return dist

def read_input_pdptw(filename):
    file_it = iter(read_elem(filename))
    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))
    # Skip two tokens (e.g. speed and an unused parameter)
    next(file_it)
    next(file_it)
    # Read depot coordinates (from the 2nd line)
    depot_x = int(next(file_it))
    depot_y = int(next(file_it))
    # Skip two tokens
    next(file_it)
    next(file_it)
    max_horizon = int(next(file_it))
    # Skip three tokens
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
        # Customer IDs are given; convert to 0-index.
        _ = int(token) - 1
        customers_x.append(int(next(file_it)))
        customers_y.append(int(next(file_it)))
        demands.append(int(next(file_it)))
        ready = int(next(file_it))
        due = int(next(file_it))
        stime = int(next(file_it))
        pick = int(next(file_it))
        delivery = int(next(file_it))
        earliest.append(ready)
        latest.append(due + stime)  # Latest finish time = due date + service time
        service_time.append(stime)
        pick_up_index.append(pick - 1)    # if pickup then -1 (after adjustment)
        delivery_index.append(delivery - 1)
        i += 1
    nb_customers = i
    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)
    return (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots,
            demands, service_time, earliest, latest, pick_up_index, delivery_index, max_horizon)

class PDPTWProblem(BaseProblem):
    """
    Pickup-and-Delivery Problem with Time Windows (PDPTW).
    
    A fleet of vehicles (with uniform capacity) must perform pickups and deliveries under time window constraints.
    Each customer has a demand, a service time, an earliest start time, and a latest finish time.
    In addition, for each pickup–delivery pair, the pickup must be performed before the corresponding delivery
    and both must occur on the same route.
    
    A candidate solution is represented as a dictionary with key "routes" mapping to a list (of length nb_trucks)
    where each element is a list of customer indices (0-indexed) representing the order in which that vehicle visits customers.
    (Customers are numbered from 0 to nb_customers–1.)
    """
    def __init__(self, instance_file: str, **kwargs):
        (self.nb_customers, self.nb_trucks, self.truck_capacity, self.dist_matrix,
         self.dist_depot, self.demands, self.service_time, self.earliest, self.latest,
         self.pick_up_index, self.delivery_index, self.max_horizon) = read_input_pdptw(instance_file)
    
    def evaluate_solution(self, solution) -> float:
        # Verify solution format.
        if not isinstance(solution, dict) or "routes" not in solution:
            return 1e15
        routes = solution["routes"]
        if not isinstance(routes, list) or len(routes) != self.nb_trucks:
            return 1e15
        # Ensure every customer is visited exactly once.
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
        
        # Enforce pickup–delivery order: for each pickup (indicated by pick_up_index == -1),
        # its corresponding delivery (given by delivery_index) must occur later in the same route.
        for i in range(self.nb_customers):
            if self.pick_up_index[i] == -1:
                found = False
                for route in routes:
                    if i in route and self.delivery_index[i] in route:
                        if route.index(i) < route.index(self.delivery_index[i]):
                            found = True
                            break
                if not found:
                    return 1e15
        
        for route in routes:
            if len(route) == 0:
                continue
            trucks_used += 1
            # Check capacity constraint.
            route_demand = sum(self.demands[i] for i in route)
            if route_demand > self.truck_capacity:
                return 1e15
            # Compute route distance: depot → first customer, inter-customer, then last customer → depot.
            r_dist = self.dist_depot[route[0]]
            for idx in range(1, len(route)):
                r_dist += self.dist_matrix[route[idx-1]][route[idx]]
            r_dist += self.dist_depot[route[-1]]
            total_distance += r_dist
            # Compute service completion times and lateness.
            # The service at the first customer starts at time = max( earliest[first], travel_time from depot )
            t_prev = max(self.earliest[route[0]], self.dist_depot[route[0]]) + self.service_time[route[0]]
            route_lateness = max(0, t_prev - self.latest[route[0]])
            for idx in range(1, len(route)):
                travel_time = self.dist_matrix[route[idx-1]][route[idx]]
                t = max(self.earliest[route[idx]], t_prev + travel_time) + self.service_time[route[idx]]
                route_lateness += max(0, t - self.latest[route[idx]])
                t_prev = t
            # Lateness for returning to depot.
            home_time = t_prev + self.dist_depot[route[-1]]
            route_lateness += max(0, home_time - self.max_horizon)
            total_lateness += route_lateness
        
        # Combine objectives lexicographically: we desire zero lateness,
        # then fewer trucks used, then lower travel distance.
        objective = total_lateness * 1e12 + trucks_used * 1e6 + total_distance
        return objective
    
    def random_solution(self):
        routes = [[] for _ in range(self.nb_trucks)]
        custs = list(range(self.nb_customers))
        random.shuffle(custs)
        for i, cust in enumerate(custs):
            routes[i % self.nb_trucks].append(cust)
        for route in routes:
            random.shuffle(route)
        return {"routes": routes}
