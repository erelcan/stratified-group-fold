import math
from ortools.linear_solver import pywraplp


def mixed_equally_partition_into_bins(id_list, weights, k):
    # Following extra mapping will allow id_list to be flexible.
    items = list(range(len(id_list)))
    bins = list(range(k))

    id_to_item_map = {}
    for i in range(len(id_list)):
        id_to_item_map[id_list[i]] = items[i]

    capacities = {}
    for c in weights:
        capacities[c] = math.ceil(sum(weights[c].values()) / k)

    solver = pywraplp.Solver.CreateSolver('SCIP')

    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in items:
        for j in bins:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # Constraints
    # Each item must be in exactly one bin.
    for i in items:
        solver.Add(sum(x[i, j] for j in bins) == 1)

    for c in weights:
        for j in bins:
            solver.Add(sum(x[(id_to_item_map[g], j)] * weights[c][g] for g in weights[c]) <= capacities[c])

    solver.Minimize(1)

    status = solver.Solve()
    
    return prepare_mixed_result(status, x, items, bins, id_list, weights)


# In the future, we may consider generalizing for train-validation-test split as well..
def mixed_weighted_split(id_list, weights, ratio, slack):
    # Following extra mapping will allow id_list to be flexible.
    items = list(range(len(id_list)))
    bins = [0, 1]

    id_to_item_map = {}
    for i in range(len(id_list)):
        id_to_item_map[id_list[i]] = items[i]
        
    upper_capacities = {}
    lower_capacities = {}
    for c in weights:
        upper_capacities[c] = math.ceil(sum(weights[c].values()) * (ratio + slack))
        lower_capacities[c] = math.ceil(sum(weights[c].values()) * (ratio - slack))

    solver = pywraplp.Solver.CreateSolver('SCIP')

    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in items:
        for j in bins:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # Constraints
    # Each item must be in exactly one bin.
    for i in items:
        solver.Add(sum(x[i, j] for j in bins) == 1)

    for c in weights:
        solver.Add(sum(x[(id_to_item_map[g], 0)] * weights[c][g] for g in weights[c]) <= upper_capacities[c])
        solver.Add(sum(x[(id_to_item_map[g], 0)] * weights[c][g] for g in weights[c]) >= lower_capacities[c])

    solver.Minimize(1)

    status = solver.Solve()

    return prepare_mixed_result(status, x, items, bins, id_list, weights)


def equally_partition_into_bins(id_list, weight_list, k, opt_type):
    # Following extra mapping will allow id_list to be flexible.
    items = list(range(len(id_list)))
    bins = list(range(k))

    # Heuristic to force bins to have equal total weight.
    capacity = math.ceil(sum(weight_list) / k)

    solver = pywraplp.Solver.CreateSolver('SCIP')

    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in items:
        for j in bins:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # Constraints
    # Each item must be in exactly one bin.
    for i in items:
        solver.Add(sum(x[i, j] for j in bins) == 1)

    # The weight total in each bin cannot exceed its capacity.
    for j in bins:
        solver.Add(sum(x[(i, j)] * weight_list[i] for i in items) <= capacity)

    if opt_type == 0:
        solver.Minimize(1)
    else:
        for j in bins[:-1]:
            solver.Add(sum(x[(i, j)] * weight_list[i] for i in items) <= sum(x[(i, j + 1)] * weight_list[i] for i in items))
        solver.Maximize(sum(x[(i, 0)] * weight_list[i] for i in items))

    status = solver.Solve()
    
    return prepare_result(status, x, items, bins, id_list, weight_list)


# In the future, we may consider generalizing for train-validation-test split as well..
def weighted_split(id_list, weight_list, ratio, slack):
    # Following extra mapping will allow id_list to be flexible.
    items = list(range(len(id_list)))
    bins = [0, 1]

    # Consider omitting ceiling~
    upper_capacity = math.ceil(sum(weight_list) * (ratio + slack))
    lower_capacity = math.ceil(sum(weight_list) * (ratio - slack))

    solver = pywraplp.Solver.CreateSolver('SCIP')

    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in items:
        for j in bins:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # Constraints
    # Each item must be in exactly one bin.
    for i in items:
        solver.Add(sum(x[i, j] for j in bins) == 1)

    solver.Add(sum(x[(i, 0)] * weight_list[i] for i in items) <= upper_capacity)
    solver.Add(sum(x[(i, 0)] * weight_list[i] for i in items) >= lower_capacity)

    solver.Maximize(sum(x[(i, 0)] * weight_list[i] for i in items))

    status = solver.Solve()

    return prepare_result(status, x, items, bins, id_list, weight_list)


def prepare_result(status, x, items, bins, id_list, weight_list):
    if status == pywraplp.Solver.OPTIMAL:
        result = {}
        for j in bins:
            result[j] = {"ids": [], "weight": 0}
            for i in items:
                if x[i, j].solution_value() > 0:
                    # Notice that we return the original ids..
                    result[j]["ids"].append(id_list[i])
                    result[j]["weight"] += weight_list[i]
        
        return result
    else:
        # In the future, return a greedy solution and raise an warning rather than exception.
        # E.g. may walk right and left of median simultaneously after sorting the weights.
        raise Exception('The problem does not have an optimal solution.')

def prepare_mixed_result(status, x, items, bins, id_list, weights):
    if status == pywraplp.Solver.OPTIMAL:
        result = {}
        for j in bins:
            class_wise_weights = {}
            for c in weights:
                class_wise_weights[c] = 0

            result[j] = {"ids": [], "weights": class_wise_weights}
            for i in items:
                if x[i, j].solution_value() > 0:
                    # Notice that we return the original ids..
                    result[j]["ids"].append(id_list[i])
                    for c in weights:
                        result[j]["weights"][c] += weights[c].get(id_list[i], 0)
        
        return result
    else:
        # In the future, return a greedy solution and raise an warning rather than exception.
        # E.g. may walk right and left of median simultaneously after sorting the weights.
        raise Exception('The problem does not have an optimal solution.')