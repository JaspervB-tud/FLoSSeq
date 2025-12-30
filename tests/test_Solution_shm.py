from reset import solution as solution
from scipy.spatial.distance import squareform
# Global imports
import numpy as np
import itertools

TOLERANCE = 6

DISTANCES_SMALL = np.array([
    [0.0, 0.1, 0.3], #0
    [0.1, 0.0, 0.5], #1
    [0.3, 0.5, 0.0]  #1
], dtype=np.float64)
CLUSTERS_SMALL = np.array([
    0, 1, 1
], dtype=np.int64)

DISTANCES_MEDIUM = np.array([
    [0.0, 0.9, 1.0,     0.7, 0.6, 0.5], #0
    [0.9, 0.0, 0.4,     0.3, 0.2, 0.1], #0
    [1.0, 0.4, 0.0,     0.5, 0.6, 0.7], #0

    [0.7, 0.3, 0.5,     0.0, 0.1, 0.2], #1
    [0.6, 0.2, 0.6,     0.1, 0.0, 0.3], #1
    [0.5, 0.1, 0.7,     0.2, 0.3, 0.0]  #1
], dtype=np.float64)
CLUSTERS_MEDIUM = np.array([
    0, 0, 0, 1, 1, 1
], dtype=np.int64)

DISTANCES_LARGE = np.array([
    [0.0, 0.2, 0.4,     0.6,    0.8, 1.0,      0.9, 0.7, 0.5, 0.3], #0
    [0.2, 0.0, 0.3,     0.5,    0.7, 0.9,      0.8, 0.6, 0.4, 0.2], #0
    [0.4, 0.3, 0.0,     0.2,    0.4, 0.6,      0.5, 0.3, 0.1, 0.2], #0

    [0.6, 0.5, 0.2,     0.0,    0.3, 0.5,      0.4, 0.2, 0.1, 0.3], #1

    [0.8, 0.7, 0.4,     0.3,    0.0, 0.2,      0.9, 0.3, 0.5, 0.7], #2
    [1.0, 0.9, 0.6,     0.5,    0.2, 0.0,      0.3, 0.5, 0.7, 0.9], #2

    [0.9, 0.8, 0.5,     0.4,    0.9, 0.3,      0.0, 0.2, 0.4, 0.6], #3
    [0.7, 0.6, 0.3,     0.2,    0.3, 0.5,      0.2, 0.0, 0.2, 0.4], #3
    [0.5, 0.4, 0.1,     0.1,    0.5, 0.7,      0.4, 0.2, 0.0, 0.2], #3
    [0.3, 0.2, 0.2,     0.3,    0.7, 0.9,      0.6, 0.4, 0.2, 0.0]  #3
])
CLUSTERS_LARGE = np.array([
    0, 0, 0, 1, 2, 2, 3, 3, 3, 3
], dtype=np.int64)

# TESTS FOR "evaluate_add" METHOD
def test_evaluate_add_small_1():
    """Test the evaluation of adding a point to a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, False, True], dtype=bool) #adding the point should improve intra and affect inter
    idx_to_add = 1
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0)
    ]
    expected_inter_changes = [
        (0, (idx_to_add, 0), 0.9),  # Cluster 0, point 1 in cluster 0, point 2 in cluster 1
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_add(idx_to_add)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)
    
def test_evaluate_add_small_2():
    """Test the evaluation of adding a point to a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, True, False], dtype=bool) #adding the point should improve intra but not affect inter
    idx_to_add = 2
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0)
    ]
    expected_inter_changes = [
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_add(idx_to_add)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_add_medium_1():
    """Test the evaluation of adding a point to a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, False, False, True, False, False], dtype=bool) #adding the point should improve intra (both for added point as well as other points)
    idx_to_add = 2
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (1, idx_to_add, 0.4),
        (idx_to_add, idx_to_add, 0.0),
    ]
    expected_inter_changes = [
        (1, (idx_to_add, 3), 0.5)
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_add(idx_to_add)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_add_medium_2():
    """Test the evaluation of adding a point to a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([False, False, True, True, False, False], dtype=bool) #adding the point improves only intra for added point
    idx_to_add = 0
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
    ]
    expected_inter_changes = [
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_add(idx_to_add)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_add_medium_3():
    """Test the evaluation of adding a point to a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([False, True, False, False, False, True], dtype=bool) #adding the point should improve multiple intra but not affect inter
    idx_to_add = 4
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (3, idx_to_add, 0.1)
    ]
    expected_inter_changes = [
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_add(idx_to_add)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_add_large_1():
    """Test the evaluation of adding a point to a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([True, False, False,   True,   True, False,    False, False, False, True], dtype=bool) #adding the point should improve multiple intra and affect inter
    idx_to_add = 1
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (2, idx_to_add, 0.3)
    ]
    expected_inter_changes = [
        (1, (idx_to_add, 3), 0.5),
        (2, (idx_to_add, 4), 0.3),
        (3, (idx_to_add, 9), 0.8)
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_add(idx_to_add)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_add_large_2():
    """Test the evaluation of adding a point to a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([False, False, True,   True,   True, False,    False, False, False, True], dtype=bool) #adding the point should improve multiple intra but not affect inter
    idx_to_add = 6
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (7, idx_to_add, 0.2)
    ]
    expected_inter_changes = [
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_add(idx_to_add)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

# TESTS FOR "accept_move" METHOD (add)
def test_accept_add_small_1():
    """Test the acceptance of adding a point to a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, False, True], dtype=bool)
    idx_to_add = 1
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_add(idx_to_add)
    solution_object.accept_move([idx_to_add], [], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_add_small_2():
    """Test the acceptance of adding a point to a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, True, False], dtype=bool)
    idx_to_add = 2
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_add(idx_to_add)
    solution_object.accept_move([idx_to_add], [], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_add_medium_1():
    """Test the acceptance of adding a point to a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, False, False, True, False, False], dtype=bool)
    idx_to_add = 2
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_add(idx_to_add)
    solution_object.accept_move([idx_to_add], [], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_add_medium_2():
    """Test the acceptance of adding a point to a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([False, False, True, True, False, False], dtype=bool)
    idx_to_add = 0
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_add(idx_to_add)
    solution_object.accept_move([idx_to_add], [], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_add_medium_3():
    """Test the acceptance of adding a point to a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([False, True, False, False, False, True], dtype=bool)
    idx_to_add = 4
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_add(idx_to_add)
    solution_object.accept_move([idx_to_add], [], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_add_large_1():
    """Test the acceptance of adding a point to a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([True, False, False, True, True, False, False, False, False, True], dtype=bool)
    idx_to_add = 1
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_add(idx_to_add)
    solution_object.accept_move([idx_to_add], [], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_add_large_2():
    """Test the acceptance of adding a point to a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([False, False, True, True, True, False, False, False, False, True], dtype=bool)
    idx_to_add = 6
    new_selection = selection.copy()
    new_selection[idx_to_add] = True

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_add(idx_to_add)
    solution_object.accept_move([idx_to_add], [], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

# TESTS FOR "evaluate_swap" METHOD
def test_evaluate_swap_small_1():
    """Test the evaluation of swapping a pair of points in a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, False, True], dtype=bool) 
    idx_to_add = 1
    idx_to_remove = 2
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (idx_to_remove, idx_to_add, 0.5)
    ]
    expected_inter_changes = [
        (0, (idx_to_add, 0), 0.9),  # Cluster 0, point 1 in cluster 0, point 2 in cluster 1
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_swap_medium_1():
    """Test the evaluation of swapping a pair of points in a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, True, False, True, False, False], dtype=bool)
    idx_to_add = 2
    idx_to_remove = 0
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (idx_to_remove, 1, 0.9)
    ]
    expected_inter_changes = [
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_swap_medium_2():
    """Test the evaluation of swapping a pair of points in a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, True, False, True, False, False], dtype=bool)
    idx_to_add = 2
    idx_to_remove = 1
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (idx_to_remove, idx_to_add, 0.4)
    ]
    expected_inter_changes = [
        (1, (idx_to_add, 3), 0.5)
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_swap_medium_3():
    """Test the evaluation of swapping a pair of points in a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, True, False, True, False, True], dtype=bool)
    idx_to_add = 4
    idx_to_remove = 5
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (idx_to_remove, 3, 0.2)
    ]
    expected_inter_changes = [
        (0, (idx_to_add, 1), 0.8)
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_swap_large_1():
    """Test the evaluation of swapping a pair of points in a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([True, False, False,   True,   True, False,    True, False, False, False], dtype=bool)
    idx_to_add = 1
    idx_to_remove = 0
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (idx_to_remove, idx_to_add, 0.2),
        (2, idx_to_add, 0.3),
    ]
    expected_inter_changes = [
        (1, (idx_to_add, 3), 0.5),
        (2, (idx_to_add, 4), 0.3),
        (3, (idx_to_add, 6), 0.2),
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_swap_large_2():
    """Test the evaluation of swapping a pair of points in a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([True, False, True,   True,   True, False,    True, False, False, False], dtype=bool)
    idx_to_add = 1
    idx_to_remove = 0
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (idx_to_remove, idx_to_add, 0.2),
    ]
    expected_inter_changes = [
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_swap_large_3():
    """Test the evaluation of swapping a pair of points in a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([False, False, True,   True,   True, False,    True, False, True, False], dtype=bool)
    idx_to_add = 7
    idx_to_remove = 6
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add, idx_to_add, 0.0),
        (idx_to_remove, idx_to_add, 0.2),
    ]
    expected_inter_changes = [
        (2, (idx_to_add, 4), 0.7),
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

# TESTS FOR "accept_move" METHOD (swap)
def test_accept_swap_small_1():
    """Test the acceptance of swapping a pair of points in a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, False, True], dtype=bool)
    idx_to_add = 1
    idx_to_remove = 2
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)
    solution_object.accept_move([idx_to_add], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_swap_medium_1():
    """Test the acceptance of swapping a pair of points in a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, True, False, True, False, False], dtype=bool)
    idx_to_add = 2
    idx_to_remove = 0
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)
    solution_object.accept_move([idx_to_add], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_swap_medium_2():
    """Test the acceptance of swapping a pair of points in a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, True, False, True, False, False], dtype=bool)
    idx_to_add = 2
    idx_to_remove = 1
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)
    solution_object.accept_move([idx_to_add], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_swap_medium_3():
    """Test the acceptance of swapping a pair of points in a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, True, False, True, False, True], dtype=bool)
    idx_to_add = 4
    idx_to_remove = 5
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)
    solution_object.accept_move([idx_to_add], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_swap_large_1():
    """Test the acceptance of swapping a pair of points in a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([True, False, False, True, True, False, True, False, False, False], dtype=bool)
    idx_to_add = 1
    idx_to_remove = 0
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)
    solution_object.accept_move([idx_to_add], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_swap_large_2():
    """Test the acceptance of swapping a pair of points in a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([True, False, True, True, True, False, True, False, False, False], dtype=bool)
    idx_to_add = 1
    idx_to_remove = 0
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)
    solution_object.accept_move([idx_to_add], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_swap_large_3():
    """Test the acceptance of swapping a pair of points in a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([False, False, True, True, True, False, True, False, True, False], dtype=bool)
    idx_to_add = 7
    idx_to_remove = 6
    new_selection = selection.copy()
    new_selection[idx_to_add] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_swap(idx_to_add, idx_to_remove)
    solution_object.accept_move([idx_to_add], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

# TESTS FOR "evaluate_swap" METHOD (double swap)
def test_evaluate_doubleswap_large_1():
    """Test the evaluation of swapping a triplet of points in a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([True, False, False,   True,   True, False,    True, False, False, False], dtype=bool)
    idx_to_add1 = 1
    idx_to_add2 = 2
    idx_to_remove = 0
    new_selection = selection.copy()
    new_selection[idx_to_add1] = True
    new_selection[idx_to_add2] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_add1, idx_to_add1, 0.0),
        (idx_to_add2, idx_to_add2, 0.0),
        (idx_to_remove, idx_to_add1, 0.2)
    ]
    expected_inter_changes = [
        (1, (idx_to_add2, 3), 0.8),
        (2, (idx_to_add2, 4), 0.6),
        (3, (idx_to_add2, 6), 0.5),
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_swap([idx_to_add1, idx_to_add2], idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

# TESTS FOR "accept_move" METHOD (double swap)
def test_accept_doubleswap_large_1():
    """Test the acceptance of swapping a triplet of points in a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([True, False, False, True, True, False, True, False, False, False], dtype=bool)
    idx_to_add1 = 1
    idx_to_add2 = 2
    idx_to_remove = 0
    new_selection = selection.copy()
    new_selection[idx_to_add1] = True
    new_selection[idx_to_add2] = True
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_swap([idx_to_add1, idx_to_add2], idx_to_remove)
    solution_object.accept_move([idx_to_add1, idx_to_add2], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

# TESTS FOR "evaluate_remove" METHOD
def test_evaluate_remove_small_1():
    """Test the evaluation of removing a point from a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, True, True], dtype=bool)
    idx_to_remove = 1
    new_selection = selection.copy()
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_remove, 2, 0.5)
    ]
    expected_inter_changes = [
        (0, (2, 0), 0.7),
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_remove(idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_remove_small_2():
    """Test the evaluation of removing a point from a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, True, True], dtype=bool)
    idx_to_remove = 2
    new_selection = selection.copy()
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_remove, 1, 0.5)
    ]
    expected_inter_changes = [
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_remove(idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_remove_medium_1():
    """Test the evaluation of removing a point from a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, True, True, True, True, True], dtype=bool)
    idx_to_remove = 2
    new_selection = selection.copy()
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_remove, 1, 0.4)
    ]
    expected_inter_changes = [
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_remove(idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

def test_evaluate_remove_large_1():
    """Test the evaluation of removing a point from a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([False, True, True,   True,   True, False,    True, False, False, False], dtype=bool)
    idx_to_remove = 1
    new_selection = selection.copy()
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)

    expected_candidate_objective, _ = calculate_objective(
        new_selection, distances, clusters, selection_cost
    )
    expected_intra_changes = [
        (idx_to_remove, 2, 0.3),
        (0, 2, 0.4),
    ]
    expected_inter_changes = [
    ]

    actual_candidate_objective, actual_intra_changes, actual_inter_changes = solution_object.evaluate_remove(idx_to_remove)

    # Compare objective values
    np.testing.assert_almost_equal(actual_candidate_objective, expected_candidate_objective, decimal=TOLERANCE)
    # Compare intra changes
    compare_intra_changes(actual_intra_changes, expected_intra_changes)
    # Compare inter changes
    compare_inter_changes(actual_inter_changes, expected_inter_changes)

# TESTS FOR "accept_move" METHOD (remove)
def test_accept_remove_small_1():
    """Test the acceptance of removing a point from a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, True, True], dtype=bool)
    idx_to_remove = 1
    new_selection = selection.copy()
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_remove(idx_to_remove)
    solution_object.accept_move([], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_remove_small_2():
    """Test the acceptance of removing a point from a small solution."""
    distances = DISTANCES_SMALL
    clusters = CLUSTERS_SMALL
    selection_cost = 0.1
    selection = np.array([True, True, True], dtype=bool)
    idx_to_remove = 2
    new_selection = selection.copy()
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_remove(idx_to_remove)
    solution_object.accept_move([], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_remove_medium_1():
    """Test the acceptance of removing a point from a medium solution."""
    distances = DISTANCES_MEDIUM
    clusters = CLUSTERS_MEDIUM
    selection_cost = 0.1
    selection = np.array([True, True, True, True, True, True], dtype=bool)
    idx_to_remove = 2
    new_selection = selection.copy()
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_remove(idx_to_remove)
    solution_object.accept_move([], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

def test_accept_remove_large_1():
    """Test the acceptance of removing a point from a large solution."""
    distances = DISTANCES_LARGE
    clusters = CLUSTERS_LARGE
    selection_cost = 0.1
    selection = np.array([False, True, True,   True,   True, False,    True, False, False, False], dtype=bool)
    idx_to_remove = 1
    new_selection = selection.copy()
    new_selection[idx_to_remove] = False

    solution_object = solution.Solution_shm(distances, clusters, selection, selection_cost=selection_cost, seed=1234)
    candidate_objective, intra_changes, inter_changes = solution_object.evaluate_remove(idx_to_remove)
    solution_object.accept_move([], [idx_to_remove], candidate_objective, intra_changes, inter_changes)

    expected_solution = solution.Solution_shm(distances, clusters, new_selection, selection_cost=selection_cost, seed=1234)

    # Test if solution objects are the same
    assert solution_object == expected_solution

# This function calculates the total objective function from scratch, as well as its components for a given solution.
def calculate_objective(selection, distances, clusters, cost_per_cluster):
    try:
        len(cost_per_cluster)
    except TypeError:
        cost_per_cluster = np.array([cost_per_cluster] * len(np.unique(clusters)), dtype=np.float64)

    objective = 0.0
    components = {
        "selection": 0.0,
        "intra": 0.0,
        "inter": 0.0,
    }

    # Assign cost for selecting
    for idx in np.where(selection)[0]:
        components["selection"] += cost_per_cluster[clusters[idx]]
        objective += cost_per_cluster[clusters[idx]]

    # Intra cluster costs
    for cluster in np.unique(clusters):
        indices_selected = np.where(selection & (clusters == cluster))[0]
        indices_nonselected = np.where(~selection & (clusters == cluster))[0]

        for idx in indices_nonselected:
            intra_cost = np.inf
            for selected_idx in indices_selected:
                intra_cost = min(intra_cost, distances[idx, selected_idx])
            components["intra"] += intra_cost
            objective += intra_cost

    # Inter cluster costs
    for cluster1, cluster2 in itertools.combinations(np.unique(clusters), 2):
        indices1 = np.where(selection & (clusters == cluster1))[0]
        indices2 = np.where(selection & (clusters == cluster2))[0]

        inter_cost = -np.inf
        for idx1 in indices1:
            for idx2 in indices2:
                inter_cost = max(inter_cost, 1.0 - distances[idx1, idx2])
        components["inter"] += inter_cost
        objective += inter_cost

    return objective, components

# This function compares the actual changes in intra cluster assignments
def compare_intra_changes(actual_changes, expected_changes):
    assert len(actual_changes) == len(expected_changes)
    for (idx1, idx2, distance), (expected_idx1, expected_idx2, expected_distance) in zip(sorted(actual_changes), sorted(expected_changes)):
        assert idx1 == expected_idx1
        assert idx2 == expected_idx2
        np.testing.assert_almost_equal(distance, expected_distance, decimal=TOLERANCE)

# This function compares the actual changes in inter cluster assignments
def compare_inter_changes(actual_changes, expected_changes):
    assert len(actual_changes) == len(expected_changes)
    for (cluster, (point_this_cluster, point_other_cluster), distance), (expected_cluster, (expected_point_this_cluster, expected_point_other_cluster), expected_distance) in zip(sorted(actual_changes), sorted(expected_changes)):
        assert cluster == expected_cluster
        assert point_this_cluster == expected_point_this_cluster
        assert point_other_cluster == expected_point_other_cluster
        np.testing.assert_almost_equal(distance, expected_distance, decimal=TOLERANCE)