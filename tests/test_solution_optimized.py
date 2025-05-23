import solution.solution_optimized as solution
from scipy.spatial.distance import squareform
# Global imports
import numpy as np
import itertools

def test_infeasible_solution_1():
    distances = np.array([[0, 0.1, 0.3], [0.1, 0, 0.5], [0.3, 0.5, 0]], dtype=np.float32)
    clusters = np.array([0, 1, 2], dtype=np.int32)

    selection_1 = np.array([False, True, True], dtype=bool) # not feasible
    solution_object_1 = solution.Solution(distances, clusters, selection=selection_1)

    selection_2 = np.array([True, True, True], dtype=bool) # feasible
    solution_object_2 = solution.Solution(distances, clusters, selection=selection_2)

    # Check if infeasible solution is correctly identified
    assert not solution_object_1.determine_feasibility()
    # Check if feasible solution is correctly identified
    assert solution_object_2.determine_feasibility()

def test_infeasible_solution_2():
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 1, 1, 2, 2], dtype=np.int32
    )
    
    selection_1 = np.array([False, True, True, False, False, False], dtype=bool) # not feasible
    solution_object_1 = solution.Solution(distances, clusters, selection=selection_1)

    selection_2 = np.array([True, False, False, True, False, True], dtype=bool) # feasible
    solution_object_2 = solution.Solution(distances, clusters, selection=selection_2)

    # Check if infeasible solution is correctly identified
    assert not solution_object_1.determine_feasibility()
    # Check if feasible solution is correctly identified
    assert solution_object_2.determine_feasibility()

def test_solution_correctness_1():
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, False, False, True, False, False], dtype=bool)
    selection_cost = 0.1
    
    expected_closest_distances_intra = np.array([0.0, 0.9, 0.8, 0.0, 0.1, 0.2], dtype=np.float32)
    expected_closest_points_intra = np.array([0, 0, 0, 3, 3, 3], dtype=np.int32)
    expected_closest_distances_inter = np.array([
        [0.0, 0.3],
        [0.3, 0.0]
        ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (0, 3)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    # Check if the distances are correctly copied
    np.testing.assert_array_equal(squareform(solution_object.distances), distances)
    # Check if the clusters are correctly copied
    np.testing.assert_array_equal(solution_object.clusters, clusters)
    # Check if the selection is correctly initiated
    np.testing.assert_array_equal(solution_object.selection, selection)
    # Check if the closest distances intra are correctly calculated
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    # Check if the closest points intra are correctly calculated
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    # Check if the closest distances inter are correctly calculated
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)
    # Check if the closest points inter are correctly calculated
    for key in expected_closest_points_inter:
        assert solution_object.closest_points_inter[key] == expected_closest_points_inter[key]
    # Check if the objective value is correctly calculated
    expected_objective_value = groundtruth_objective_value(selection, clusters, distances, selection_cost)
    assert solution_object.objective == expected_objective_value

def test_evaluate_add_1():
    # Small instance, 6 points, 2 clusters
    # In this test, the point added as well as other points have associated changes
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, False, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Selecting point at index 1 (adding point 1 to the solution)
    # Expected output for adding index 1:
    expected_selection = np.array([True, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(1), np.float32(0.0)),
        (np.int64(2), np.float32(0.4))
    ]
    expected_inter_changes = [
        (np.int32(1), np.float32(0.7), np.int64(3))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_add(1)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_add_2():
    # Small instance, 6 points, 2 clusters
    # In this test, only the point added has associated changes
    distances = np.array([
        [0.0, 0.9, 1.0, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [1.0, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([False, True, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Adding point 3 (index 2) to the solution
    expected_selection = np.array([False, True, True, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(2), np.float32(0.0))
    ]
    expected_inter_changes = [
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_add(2)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_add_3():
    distances = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0, 0.2, 0.4, 0.6, 0.5, 0.3, 0.1, 0.2], #0
        [0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4, 0.3, 0.0, 0.2, 0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.7, 0.9], #2
        [0.9, 0.8, 0.5, 0.4, 0.1, 0.3, 0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3, 0.2, 0.3, 0.5, 0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1, 0.1, 0.5, 0.7, 0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2, 0.3, 0.7, 0.9, 0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, False, False, True, True, False, True, False, True, False], dtype=bool)
    selection_cost = 0.5

    # Adding point 3 (index 2) to the solution
    expected_selection = np.array([True, False, True, True, True, False, True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(2), np.float32(0.0))
    ]
    expected_inter_changes = [
        (np.int32(1), np.float32(0.8), np.int64(3)),
        (np.int32(2), np.float32(0.6), np.int64(4)),
        (np.int32(3), np.float32(0.9), np.int64(8))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_add(2)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_accept_add_1():
    # Small instance, 6 points, 2 clusters
    # In this test, the point added as well as other points have associated changes
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, False, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Selecting point at index 1 (adding point 1 to the solution)
    # Expected output for adding index 1:
    expected_selection = np.array([True, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)     
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5}
    }
    expected_selection_per_cluster = {
        0: {0, 1},
        1: {3}
    }
    expected_nonselection_per_cluster = {
        0: {2},
        1: {4, 5}
    }
    expected_closest_distances_intra = np.array(
        [0.0, 0.0, 0.4, 0.0, 0.1, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [0, 1, 1, 3, 3, 3], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.7],
        [0.7, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 3)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_add(1)
    solution_object.accept_add(1, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_add_2():
    # Small instance, 6 points, 2 clusters
    # In this test, only the point added has associated changes
    distances = np.array([
        [0.0, 0.9, 1.0, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [1.0, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([False, True, True, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Adding point 6 (index 5) to the solution
    expected_selection = np.array([False, True, True, True, False, True], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5}
    }
    expected_selection_per_cluster = {
        0: {1, 2},
        1: {3, 5}
    }
    expected_nonselection_per_cluster = {
        0: {0},
        1: {4}
    }
    expected_closest_distances_intra = np.array(
        [0.9, 0.0, 0.0, 0.0, 0.1, 0.0], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [1, 1, 2, 3, 3, 5], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.9],
        [0.9, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 5)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_add(5)
    solution_object.accept_add(5, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_add_3():
    distances = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0, 0.2, 0.4, 0.6, 0.5, 0.3, 0.1, 0.2], #0
        [0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4, 0.3, 0.0, 0.2, 0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.7, 0.9], #2
        [0.9, 0.8, 0.5, 0.4, 0.1, 0.3, 0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3, 0.2, 0.3, 0.5, 0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1, 0.1, 0.5, 0.7, 0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2, 0.3, 0.7, 0.9, 0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, False, False, True, True, False, True, False, True, False], dtype=bool)
    selection_cost = 0.5

    # Adding point 3 (index 2) to the solution
    expected_selection = np.array([True, False, True, True, True, False, True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3},
        2: {4, 5},
        3: {6, 7, 8, 9}
    }
    expected_selection_per_cluster = {
        0: {0, 2},
        1: {3},
        2: {4},
        3: {6, 8}
    }
    expected_nonselection_per_cluster = {
        0: {1},
        1: set(),
        2: {5},
        3: {7, 9}
    }
    expected_closest_distances_intra = np.array(
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [0, 0, 2, 3, 4, 4, 6, 6, 8, 8], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.8, 0.6, 0.9],
        [0.8, 0.0, 0.7, 0.9],
        [0.6, 0.7, 0.0, 0.9],
        [0.9, 0.9, 0.9, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (2, 3),
        (0, 2): (2, 4),
        (0, 3): (2, 8),
        (1, 2): (3, 4),
        (1, 3): (3, 8),
        (2, 3): (4, 6)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_add(2)
    solution_object.accept_add(2, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_evaluate_swap_1():
    # Small instance, 6 points, 2 clusters, test when swapping out uniquely selected point
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, False, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Swapping point 1 (index 0) with point 2 (index 1)
    expected_selection = np.array([False, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(0), np.int64(1), np.float32(0.9)),
        (np.int64(1), np.int64(1), np.float32(0.0)),
        (np.int64(2), np.int64(1), np.float32(0.4))
    ]
    expected_inter_changes = [
        (np.int32(1), (np.int64(1), np.int64(3)), np.float32(0.7))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_swap(1, 0)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_swap_2():
    # Small instance, 6 points, 2 clusters, swap should only affect cluster of swapped points
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, True, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Swapping point 1 (index 0) with point 3 (index 2)
    expected_selection = np.array([False, True, True, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(0), np.int64(2), np.float32(0.8)),
        (np.int64(2), np.int64(2), np.float32(0.0))
    ]
    expected_inter_changes = [
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_swap(2, 0)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_swap_3():
    # Larger instance, 10 points, 4 clusters, test when swapping out uniquely selected point
    distances = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0, 0.2, 0.4, 0.6, 0.5, 0.3, 0.1, 0.2], #0
        [0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4, 0.3, 0.0, 0.2, 0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.7, 0.9], #2
        [0.9, 0.8, 0.5, 0.4, 0.1, 0.3, 0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3, 0.2, 0.3, 0.5, 0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1, 0.1, 0.5, 0.7, 0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2, 0.3, 0.7, 0.9, 0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, False, False, True, True, False, True, False, True, False], dtype=bool)
    selection_cost = 0.5

    # Swapping point 1 (index 0) with point 3 (index 2)
    expected_selection = np.array([False, False, True, True, True, False, True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(0), np.int64(2), np.float32(0.4)),
        (np.int64(1), np.int64(2), np.float32(0.3)),
        (np.int64(2), np.int64(2), np.float32(0.0))
    ]
    expected_inter_changes = [
        (np.int32(1), (np.int64(2), np.int64(3)), np.float32(0.8)),
        (np.int32(2), (np.int64(2), np.int64(4)), np.float32(0.6)),
        (np.int32(3), (np.int64(2), np.int64(8)), np.float32(0.9))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_swap(2, 0)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_swap_4():
    # Larger instance, 10 points, 3 clusters, test when swapping out NON-uniquely selected point
    distances = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0, 0.2, 0.4, 0.6, 0.5, 0.3, 0.1, 0.2], #0
        [0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4, 0.3, 0.0, 0.2, 0.1, 0.3, 0.5, 0.7], #1
        [1.0, 0.9, 0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.7, 0.9], #1
        [0.9, 0.8, 0.5, 0.4, 0.1, 0.3, 0.0, 0.2, 0.4, 0.6], #2
        [0.7, 0.6, 0.3, 0.2, 0.3, 0.5, 0.2, 0.0, 0.2, 0.4], #2
        [0.5, 0.4, 0.1, 0.1, 0.5, 0.7, 0.4, 0.2, 0.0, 0.2], #2
        [0.3, 0.2, 0.2, 0.3, 0.7, 0.9, 0.6, 0.4, 0.2, 0.0]  #2
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32
    )

    selection = np.array([True, True, False, True, True, False, True, False, True, False], dtype=bool)
    #                       0     1     2      3     4     5     6     7      8     9
    selection_cost = 0.5

    # Swapping point 2 (index 1) with point 3 (index 2)
    expected_selection = np.array([True, False, True, True, True, False, True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(1), np.int64(0), np.float32(0.2)),
        (np.int64(2), np.int64(2), np.float32(0.0))
    ]
    expected_inter_changes = [
        (np.int64(1), (np.int64(2), np.int64(3)), np.float32(0.8)),
        (np.int64(2), (np.int64(2), np.int64(8)), np.float32(0.9))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_swap(2, 1)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_accept_swap_1():
    # Small instance, 6 points, 2 clusters, test when swapping out uniquely selected point
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, False, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Swapping point 1 (index 0) with point 2 (index 1)
    expected_selection = np.array([False, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5}
    }
    expected_selection_per_cluster = {
        0: {1},
        1: {3}
    }
    expected_nonselection_per_cluster = {
        0: {0, 2},
        1: {4, 5}
    }
    expected_closest_distances_intra = np.array(
        [0.9, 0.0, 0.4, 0.0, 0.1, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [1, 1, 1, 3, 3, 3], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.7],
        [0.7, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 3)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_swap(1, 0)
    solution_object.accept_swap(1, 0, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_swap_2():
    # Small instance, 6 points, 2 clusters, swap should only affect cluster of swapped points
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, True, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Swapping point 1 (index 0) with point 3 (index 2)
    expected_selection = np.array([False, True, True, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5}
    }
    expected_selection_per_cluster = {
        0: {1, 2},
        1: {3}
    }
    expected_nonselection_per_cluster = {
        0: {0},
        1: {4, 5}
    }
    expected_closest_distances_intra = np.array(
        [0.8, 0.0, 0.0, 0.0, 0.1, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [2, 1, 2, 3, 3, 3], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.7],
        [0.7, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 3)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_swap(2, 0)
    solution_object.accept_swap(2, 0, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_swap_3():
    # Larger instance, 10 points, 4 clusters, test when swapping out uniquely selected point
    distances = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0, 0.2, 0.4, 0.6, 0.5, 0.3, 0.1, 0.2], #0
        [0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4, 0.3, 0.0, 0.2, 0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.7, 0.9], #2
        [0.9, 0.8, 0.5, 0.4, 0.1, 0.3, 0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3, 0.2, 0.3, 0.5, 0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1, 0.1, 0.5, 0.7, 0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2, 0.3, 0.7, 0.9, 0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, False, False, True, True, False, True, False, True, False], dtype=bool)
    selection_cost = 0.5

    # Swapping point 1 (index 0) with point 3 (index 2)
    expected_selection = np.array([False, False, True, True, True, False, True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3},
        2: {4, 5},
        3: {6, 7, 8, 9}
    }
    expected_selection_per_cluster = {
        0: {2},
        1: {3},
        2: {4},
        3: {6, 8}
    }
    expected_nonselection_per_cluster = {
        0: {0, 1},
        1: set(),
        2: {5},
        3: {7,9}
    }
    expected_closest_distances_intra = np.array(
        [0.4, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [2, 2, 2, 3, 4, 4, 6, 6, 8, 8], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.8, 0.6, 0.9],
        [0.8, 0.0, 0.7, 0.9],
        [0.6, 0.7, 0.0, 0.9],
        [0.9, 0.9, 0.9, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (2, 3),
        (0, 2): (2, 4),
        (0, 3): (2, 8),
        (1, 2): (3, 4),
        (1, 3): (3, 8),
        (2, 3): (4, 6)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_swap(2, 0)
    solution_object.accept_swap(2, 0, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_swap_4():
    # Larger instance, 10 points, 3 clusters, test when swapping out NON-uniquely selected point
    distances = np.array([
        [0.0, 0.2, 0.4,     0.6, 0.8, 1.0,      0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3,     0.5, 0.7, 0.9,      0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0,     0.2, 0.4, 0.6,      0.5, 0.3, 0.1, 0.2], #0

        [0.6, 0.5, 0.2,     0.0, 0.3, 0.5,      0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4,     0.3, 0.0, 0.2,      0.1, 0.3, 0.5, 0.7], #1
        [1.0, 0.9, 0.6,     0.5, 0.2, 0.0,      0.3, 0.5, 0.7, 0.9], #1

        [0.9, 0.8, 0.5,     0.4, 0.1, 0.3,      0.0, 0.2, 0.4, 0.6], #2
        [0.7, 0.6, 0.3,     0.2, 0.3, 0.5,      0.2, 0.0, 0.2, 0.4], #2
        [0.5, 0.4, 0.1,     0.1, 0.5, 0.7,      0.4, 0.2, 0.0, 0.2], #2
        [0.3, 0.2, 0.2,     0.3, 0.7, 0.9,      0.6, 0.4, 0.2, 0.0]  #2
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32
    )

    selection = np.array([True, True, False, True, True, False, True, False, True, False], dtype=bool)
    #                       0     1     2      3     4     5     6     7      8     9
    selection_cost = 0.5

    # Swapping point 2 (index 1) with point 3 (index 2)
    expected_selection = np.array([True, False, True, True, True, False, True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5},
        2: {6, 7, 8, 9}
    }
    expected_selection_per_cluster = {
        0: {0, 2},
        1: {3, 4},
        2: {6, 8}
    }
    expected_nonselection_per_cluster = {
        0: {1},
        1: {5},
        2: {7, 9}
    }
    expected_closest_distances_intra = np.array(
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [0, 0, 2, 3, 4, 4, 6, 6, 8, 8], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.8, 0.9],
        [0.8, 0.0, 0.9],
        [0.9, 0.9, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (2, 3),
        (0, 2): (2, 8),
        (1, 2): (3, 8)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_swap(2, 1)
    solution_object.accept_swap(2, 1, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_evaluate_doubleswap_1():
    # Small instance, 6 points, 2 clusters, test when swapping out uniquely selected point
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, False, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Swapping point 1 (index 0) with point 2 (index 1) and point 3 (index 2)
    expected_selection = np.array([False, True, True, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(0), np.int64(2), np.float32(0.8)),
        (np.int64(1), np.int64(1), np.float32(0.0)),
        (np.int64(2), np.int64(2), np.float32(0.0))
    ]
    expected_inter_changes = [
        (np.int32(1), (np.int64(1), np.int64(3)), np.float32(0.7))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_doubleswap((1,2), 0)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_doubleswap_2():
    # Small instance, 7 points, 2 clusters, double swap should only affect cluster of swapped points
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.9], #0
        [0.9, 0.0, 0.4, 0.3, 0.1, 0.1, 0.1], #0
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7, 0.9], #0
        [0.7, 0.3, 0.5, 0.0, 0.2, 0.2, 0.9], #0
        [0.6, 0.1, 0.6, 0.2, 0.0, 0.3, 0.9], #1
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0, 0.9], #1
        [0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.0]  #0
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 0, 1, 1, 0], dtype=np.int32
    )

    selection = np.array([False, True, False, True, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Swapping point 4 (index 3) with point 1 (index 0) and point 3 (index 2)
    expected_selection = np.array([True, True, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(0), np.int64(0), np.float32(0.0)),
        (np.int64(2), np.int64(2), np.float32(0.0)),
        (np.int64(3), np.int64(1), np.float32(0.3))
    ]
    expected_inter_changes = [
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_doubleswap((0,2), 3)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_doubleswap_3():
    # Larger instance, 10 points, 4 clusters, test when swapping out uniquely selected point
    distances = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0, 0.2, 0.4, 0.6, 0.5, 0.3, 0.1, 0.2], #0
        [0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4, 0.3, 0.0, 0.2, 0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.7, 0.9], #2
        [0.9, 0.8, 0.5, 0.4, 0.1, 0.3, 0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3, 0.2, 0.3, 0.5, 0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1, 0.1, 0.5, 0.7, 0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2, 0.3, 0.7, 0.9, 0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, False, False, True, True, False, True, False, True, False], dtype=bool)
    #                       0     1     2      3     4      5      6     7      8     9
    selection_cost = 0.5

    # Swapping point 7 (index 6) with point 8 (index 7) and point 10 (index 9)
    expected_selection = np.array([True, False, False, True, True, False, False, True, True, True], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(6), np.int64(7), np.float32(0.2)),
        (np.int64(7), np.int64(7), np.float32(0.0)),
        (np.int64(9), np.int64(9), np.float32(0.0))
    ]
    expected_inter_changes = [
        (np.int32(0), (np.int64(0), np.int64(9)), np.float32(0.7)),
        (np.int32(2), (np.int64(4), np.int64(7)), np.float32(0.7))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_doubleswap((7,9), 6)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_doubleswap_4():
    # Larger instance, 10 points, 3 clusters, test when swapping out NON-uniquely selected point
    distances = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0, 0.2, 0.4, 0.6, 0.5, 0.3, 0.1, 0.2], #0
        [0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4, 0.3, 0.0, 0.2, 0.1, 0.3, 0.5, 0.7], #1
        [1.0, 0.9, 0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.7, 0.9], #1
        [0.9, 0.8, 0.5, 0.4, 0.1, 0.3, 0.0, 0.2, 0.4, 0.6], #2
        [0.7, 0.6, 0.3, 0.2, 0.3, 0.5, 0.2, 0.0, 0.2, 0.4], #2
        [0.5, 0.4, 0.1, 0.1, 0.5, 0.7, 0.4, 0.2, 0.0, 0.2], #2
        [0.3, 0.2, 0.2, 0.3, 0.7, 0.9, 0.6, 0.4, 0.2, 0.0]  #2
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32
    )

    selection = np.array([True, True, False, True, True, False, True, False, False, False], dtype=bool)
    #                       0     1     2      3     4     5     6     7       8      9
    selection_cost = 0.5

    # Swapping point 7 (index 6) with points 8 (index 7) and 9 (index 8)
    expected_selection = np.array([True, True, False, True, True, False, False, True, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(6), np.int64(7), np.float32(0.2)),
        (np.int64(7), np.int64(7), np.float32(0.0)),
        (np.int64(8), np.int64(8), np.float32(0.0)),
        (np.int64(9), np.int64(8), np.float32(0.2))
    ]
    expected_inter_changes = [
        (np.int64(0), (np.int64(1), np.int64(8)), np.float32(0.6)),
        (np.int64(1), (np.int64(3), np.int64(8)), np.float32(0.9))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_doubleswap((7,8), 6)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_accept_doubleswap_1():
    # Small instance, 6 points, 2 clusters, test when swapping out uniquely selected point
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, False, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Swapping point 1 (index 0) with point 2 (index 1) and point 3 (index 2)
    expected_selection = np.array([False, True, True, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5}
    }
    expected_selection_per_cluster = {
        0: {1, 2},
        1: {3}
    }
    expected_nonselection_per_cluster = {
        0: {0},
        1: {4, 5}
    }
    expected_closest_distances_intra = np.array(
        [0.8, 0.0, 0.0, 0.0, 0.1, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [2, 1, 2, 3, 3, 3], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.7],
        [0.7, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 3)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_doubleswap((1,2), 0)
    solution_object.accept_doubleswap((1,2), 0, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_doubleswap_2():
    # Small instance, 7 points, 2 clusters, double swap should only affect cluster of swapped points
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.9], #0
        [0.9, 0.0, 0.4, 0.3, 0.1, 0.1, 0.1], #0
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7, 0.9], #0
        [0.7, 0.3, 0.5, 0.0, 0.2, 0.2, 0.9], #0
        [0.6, 0.1, 0.6, 0.2, 0.0, 0.3, 0.9], #1
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0, 0.9], #1
        [0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.0]  #0
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 0, 1, 1, 0], dtype=np.int32
    )

    selection = np.array([False, True, False, True, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Swapping point 4 (index 3) with point 1 (index 0) and point 3 (index 2)
    expected_selection = np.array([True, True, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2, 3, 6},
        1: {4, 5}
    }
    expected_selection_per_cluster = {
        0: {0, 1, 2},
        1: {4}
    }
    expected_nonselection_per_cluster = {
        0: {3, 6},
        1: {5}
    }
    expected_closest_distances_intra = np.array(
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.1], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [0, 1, 2, 1, 4, 4, 1], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.9],
        [0.9, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 4)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_doubleswap((0,2), 3)
    solution_object.accept_doubleswap((0,2), 3, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_doubleswap_3():
    # Larger instance, 10 points, 4 clusters, test when swapping out uniquely selected point
    distances = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0, 0.2, 0.4, 0.6, 0.5, 0.3, 0.1, 0.2], #0
        [0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4, 0.3, 0.0, 0.2, 0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6, 0.5, 0.2, 0.0, 0.3, 0.5, 0.7, 0.9], #2
        [0.9, 0.8, 0.5, 0.4, 0.1, 0.3, 0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3, 0.2, 0.3, 0.5, 0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1, 0.1, 0.5, 0.7, 0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2, 0.3, 0.7, 0.9, 0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, False, False, True, True, False, True, False, True, False], dtype=bool)
    #                       0     1     2      3     4      5      6     7      8     9
    selection_cost = 0.5

    # Swapping point 7 (index 6) with point 8 (index 7) and point 10 (index 9)
    expected_selection = np.array([True, False, False, True, True, False, False, True, True, True], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3},
        2: {4, 5},
        3: {6, 7, 8, 9}
    }
    expected_selection_per_cluster = {
        0: {0},
        1: {3},
        2: {4},
        3: {7, 8, 9}
    }
    expected_nonselection_per_cluster = {
        0: {1, 2},
        1: set(),
        2: {5},
        3: {6}
    }
    expected_closest_distances_intra = np.array(
        [0.0, 0.2, 0.4, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [0, 0, 0, 3, 4, 4, 7, 7, 8, 9], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.4, 0.2, 0.7],
        [0.4, 0.0, 0.7, 0.9],
        [0.2, 0.7, 0.0, 0.7],
        [0.7, 0.9, 0.7, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (0, 3),
        (0, 2): (0, 4),
        (0, 3): (0, 9),
        (1, 2): (3, 4),
        (1, 3): (3, 8),
        (2, 3): (4, 7)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_doubleswap((7,9), 6)
    solution_object.accept_doubleswap((7,9), 6, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_almost_equal(solution_object.closest_distances_intra, expected_closest_distances_intra, decimal=5)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_almost_equal(solution_object.closest_distances_inter, expected_closest_distances_inter, decimal=5)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_doubleswap_4():
    # Larger instance, 10 points, 3 clusters, test when swapping out NON-uniquely selected point
    distances = np.array([
        [0.0, 0.2, 0.4,     0.6, 0.8, 1.0,      0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3,     0.5, 0.7, 0.9,      0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0,     0.2, 0.4, 0.6,      0.5, 0.3, 0.1, 0.2], #0

        [0.6, 0.5, 0.2,     0.0, 0.3, 0.5,      0.4, 0.2, 0.1, 0.3], #1
        [0.8, 0.7, 0.4,     0.3, 0.0, 0.2,      0.1, 0.3, 0.5, 0.7], #1
        [1.0, 0.9, 0.6,     0.5, 0.2, 0.0,      0.3, 0.5, 0.7, 0.9], #1

        [0.9, 0.8, 0.5,     0.4, 0.1, 0.3,      0.0, 0.2, 0.4, 0.6], #2
        [0.7, 0.6, 0.3,     0.2, 0.3, 0.5,      0.2, 0.0, 0.2, 0.4], #2
        [0.5, 0.4, 0.1,     0.1, 0.5, 0.7,      0.4, 0.2, 0.0, 0.2], #2
        [0.3, 0.2, 0.2,     0.3, 0.7, 0.9,      0.6, 0.4, 0.2, 0.0]  #2
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32
    )

    selection = np.array([True, True, False, True, True, False, True, False, False, False], dtype=bool)
    #                       0     1     2      3     4     5     6     7       8      9
    selection_cost = 0.5

    # Swapping point 7 (index 6) with points 8 (index 7) and 9 (index 8)
    expected_selection = np.array([True, True, False, True, True, False, False, True, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5},
        2: {6, 7, 8, 9}
    }
    expected_selection_per_cluster = {
        0: {0, 1},
        1: {3, 4},
        2: {7, 8}
    }
    expected_nonselection_per_cluster = {
        0: {2},
        1: {5},
        2: {6, 9}
    }
    expected_closest_distances_intra = np.array(
        [0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [0, 1, 1, 3, 4, 4, 7, 7, 8, 8], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.5, 0.6],
        [0.5, 0.0, 0.9],
        [0.6, 0.9, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 3),
        (0, 2): (1, 8),
        (1, 2): (3, 8)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_doubleswap((7,8), 6)
    solution_object.accept_doubleswap((7,8), 6, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_evaluate_remove_1():
    # Small instance, 6 points, 2 clusters
    # In this test, the point removed as well as other points have associated changes
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, True, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Removing point at index 1 (removing point 2 from the solution)
    # Expected output for removing index 1:
    expected_selection = np.array([True, False, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(1), np.int64(0), np.float32(0.9)),
        (np.int64(2), np.int64(0), np.float32(0.8))
    ]
    expected_inter_changes = [
        (np.int32(1), (np.int64(0), np.int64(3)), np.float32(0.3))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(1)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_remove_2():
    # Small instance, 6 points, 2 clusters
    # In this test, only the point removed will have associated changes
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, True, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Removing point at index 0 (removing point 1 from the solution)
    # Expected output for removing index 0:
    expected_selection = np.array([False, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(0), np.int64(1), np.float32(0.9))
    ]
    expected_inter_changes = [
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(0)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_remove_3():
    # Small instance, 6 points, 2 clusters
    # In this test, only the point removed will have associated changes and the closest point for the other cluster
    distances = np.array([
        [0.0, 0.9, 0.8, 0.1, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.1, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, True, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Removing point at index 0 (removing point 1 from the solution)
    # Expected output for removing index 0:
    expected_selection = np.array([False, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (np.int64(0), np.int64(1), np.float32(0.9))
    ]
    expected_inter_changes = [
        (np.int32(1), (np.int64(1), np.int64(3)), np.float32(0.7))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(0)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_remove_4():
    distances = np.array([
        [0.0, 0.2, 0.4,     0.6,    0.8, 1.0,   0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3,     0.5,    0.7, 0.9,   0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0,     0.2,    0.4, 0.6,   0.5, 0.3, 0.1, 0.2], #0

        [0.6, 0.5, 0.2,     0.0,    0.3, 0.5,   0.4, 0.2, 0.1, 0.3], #1

        [0.8, 0.7, 0.4,     0.3,    0.0, 0.2,   0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6,     0.5,    0.2, 0.0,   0.3, 0.5, 0.7, 0.9], #2

        [0.9, 0.8, 0.5,     0.4,    0.1, 0.3,   0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3,     0.2,    0.3, 0.5,   0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1,     0.1,    0.5, 0.7,   0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2,     0.3,    0.7, 0.9,   0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, True, False,     True,      True, False,    True, False, True, False], dtype=bool)
    selection_cost = 0.5

    # Removing point 1 (index 0) to the solution
    expected_selection = np.array([False, True, False,  True,   True, False,    True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (0, 1, np.float32(0.2))
    ]
    expected_inter_changes = [
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(0)

    assert sorted(inter_changes) == sorted(expected_inter_changes)
    assert sorted(intra_changes) == sorted(expected_intra_changes)
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_evaluate_remove_5():
    distances = np.array([
        [0.0, 0.2, 0.4,     0.5,    0.8, 1.0,   0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3,     0.6,    0.7, 0.9,   0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0,     0.2,    0.4, 0.6,   0.5, 0.3, 0.1, 0.2], #0

        [0.5, 0.6, 0.2,     0.0,    0.3, 0.5,   0.4, 0.2, 0.1, 0.3], #1

        [0.8, 0.7, 0.4,     0.3,    0.0, 0.2,   0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6,     0.5,    0.2, 0.0,   0.3, 0.5, 0.7, 0.9], #2

        [0.9, 0.8, 0.5,     0.4,    0.1, 0.3,   0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3,     0.2,    0.3, 0.5,   0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1,     0.1,    0.5, 0.7,   0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2,     0.3,    0.7, 0.9,   0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, True, False,     True,      True, False,    True, False, True, False], dtype=bool)
    selection_cost = 0.5

    # Removing point 1 (index 0) to the solution
    expected_selection = np.array([False, True, False,  True,   True, False,    True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_intra_changes = [
        (0, 1, np.float32(0.2))
    ]
    expected_inter_changes = [
        (1, (1, 3), np.float32(0.4))
    ]

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(0)

    compare_tuples(sorted(inter_changes), sorted(expected_inter_changes))
    compare_tuples(sorted(intra_changes), sorted(expected_intra_changes))
    np.testing.assert_almost_equal(new_objective_value, expected_objective_value, decimal=5)

def test_accept_remove_1():
    # Small instance, 6 points, 2 clusters
    # In this test, the point removed as well as other points have associated changes
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, True, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Removing point at index 1 (removing point 2 from the solution)
    # Expected output for removing index 1:
    expected_selection = np.array([True, False, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5}
    }
    expected_selection_per_cluster = {
        0: {0},
        1: {3}
    }
    expected_nonselection_per_cluster = {
        0: {1, 2},
        1: {4, 5}
    }
    expected_closest_distances_intra = np.array(
        [0.0, 0.9, 0.8, 0.0, 0.1, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [0, 0, 0, 3, 3, 3], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.3],
        [0.3, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (0, 3)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(1)
    solution_object.accept_remove(1, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_remove_2():
    # Small instance, 6 points, 2 clusters
    # In this test, only the point removed will have associated changes
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, True, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Removing point at index 0 (removing point 1 from the solution)
    # Expected output for removing index 0:
    expected_selection = np.array([False, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5}
    }
    expected_selection_per_cluster = {
        0: {1},
        1: {3}
    }
    expected_nonselection_per_cluster = {
        0: {0, 2},
        1: {4, 5}
    }
    expected_closest_distances_intra = np.array(
        [0.9, 0.0, 0.4, 0.0, 0.1, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [1, 1, 1, 3, 3, 3], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.7],
        [0.7, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 3)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(0)
    solution_object.accept_remove(0, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_remove_3():
    # Small instance, 6 points, 2 clusters
    # In this test, only the point removed will have associated changes and the closest point for the other cluster
    distances = np.array([
        [0.0, 0.9, 0.8, 0.1, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.1, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, True, False, True, False, False], dtype=bool)
    selection_cost = 0.1

    # Removing point at index 0 (removing point 1 from the solution)
    # Expected output for removing index 0:
    expected_selection = np.array([False, True, False, True, False, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3, 4, 5}
    }
    expected_selection_per_cluster = {
        0: {1},
        1: {3}
    }
    expected_nonselection_per_cluster = {
        0: {0, 2},
        1: {4, 5}
    }
    expected_closest_distances_intra = np.array(
        [0.9, 0.0, 0.4, 0.0, 0.1, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [1, 1, 1, 3, 3, 3], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.7],
        [0.7, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 3)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(0)
    solution_object.accept_remove(0, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_remove_4():
    distances = np.array([
        [0.0, 0.2, 0.4,     0.6,    0.8, 1.0,   0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3,     0.5,    0.7, 0.9,   0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0,     0.2,    0.4, 0.6,   0.5, 0.3, 0.1, 0.2], #0

        [0.6, 0.5, 0.2,     0.0,    0.3, 0.5,   0.4, 0.2, 0.1, 0.3], #1

        [0.8, 0.7, 0.4,     0.3,    0.0, 0.2,   0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6,     0.5,    0.2, 0.0,   0.3, 0.5, 0.7, 0.9], #2

        [0.9, 0.8, 0.5,     0.4,    0.1, 0.3,   0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3,     0.2,    0.3, 0.5,   0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1,     0.1,    0.5, 0.7,   0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2,     0.3,    0.7, 0.9,   0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, True, False,     True,      True, False,    True, False, True, False], dtype=bool)
    selection_cost = 0.5

    # Removing point 1 (index 0) to the solution
    expected_selection = np.array([False, True, False,  True,   True, False,    True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3},
        2: {4, 5},
        3: {6, 7, 8, 9}
    }
    expected_selection_per_cluster = {
        0: {1},
        1: {3},
        2: {4},
        3: {6, 8}
    }
    expected_nonselection_per_cluster = {
        0: {0, 2},
        1: set(),
        2: {5},
        3: {7, 9}
    }
    expected_closest_distances_intra = np.array(
        [0.2, 0.0, 0.3, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [1, 1, 1, 3, 4, 4, 6, 6, 8, 8], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.5, 0.3, 0.6],
        [0.5, 0.0, 0.7, 0.9],
        [0.3, 0.7, 0.0, 0.9],
        [0.6, 0.9, 0.9, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 3),
        (0, 2): (1, 4),
        (0, 3): (1, 8),
        (1, 2): (3, 4),
        (1, 3): (3, 8),
        (2, 3): (4, 6)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(0)
    solution_object.accept_remove(0, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

def test_accept_remove_5():
    distances = np.array([
        [0.0, 0.2, 0.4,     0.5,    0.8, 1.0,   0.9, 0.7, 0.5, 0.3], #0
        [0.2, 0.0, 0.3,     0.6,    0.7, 0.9,   0.8, 0.6, 0.4, 0.2], #0
        [0.4, 0.3, 0.0,     0.2,    0.4, 0.6,   0.5, 0.3, 0.1, 0.2], #0

        [0.5, 0.6, 0.2,     0.0,    0.3, 0.5,   0.4, 0.2, 0.1, 0.3], #1

        [0.8, 0.7, 0.4,     0.3,    0.0, 0.2,   0.1, 0.3, 0.5, 0.7], #2
        [1.0, 0.9, 0.6,     0.5,    0.2, 0.0,   0.3, 0.5, 0.7, 0.9], #2

        [0.9, 0.8, 0.5,     0.4,    0.1, 0.3,   0.0, 0.2, 0.4, 0.6], #3
        [0.7, 0.6, 0.3,     0.2,    0.3, 0.5,   0.2, 0.0, 0.2, 0.4], #3
        [0.5, 0.4, 0.1,     0.1,    0.5, 0.7,   0.4, 0.2, 0.0, 0.2], #3
        [0.3, 0.2, 0.2,     0.3,    0.7, 0.9,   0.6, 0.4, 0.2, 0.0]  #3
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32
    )

    selection = np.array([True, True, False,     True,      True, False,    True, False, True, False], dtype=bool)
    selection_cost = 0.5

    # Removing point 1 (index 0) to the solution
    expected_selection = np.array([False, True, False,  True,   True, False,    True, False, True, False], dtype=bool)
    expected_objective_value = groundtruth_objective_value(expected_selection, clusters, distances, selection_cost)
    expected_points_per_cluster = {
        0: {0, 1, 2},
        1: {3},
        2: {4, 5},
        3: {6, 7, 8, 9}
    }
    expected_selection_per_cluster = {
        0: {1},
        1: {3},
        2: {4},
        3: {6, 8}
    }
    expected_nonselection_per_cluster = {
        0: {0, 2},
        1: set(),
        2: {5},
        3: {7, 9}
    }
    expected_closest_distances_intra = np.array(
        [0.2, 0.0, 0.3, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2], dtype=np.float32
    )
    expected_closest_points_intra = np.array(
        [1, 1, 1, 3, 4, 4, 6, 6, 8, 8], dtype=np.int32
    )
    expected_closest_distances_inter = np.array([
        [0.0, 0.4, 0.3, 0.6],
        [0.4, 0.0, 0.7, 0.9],
        [0.3, 0.7, 0.0, 0.9],
        [0.6, 0.9, 0.9, 0.0]
    ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (1, 3),
        (0, 2): (1, 4),
        (0, 3): (1, 8),
        (1, 2): (3, 4),
        (1, 3): (3, 8),
        (2, 3): (4, 6)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    new_objective_value, intra_changes, inter_changes = solution_object.evaluate_remove(0)
    solution_object.accept_remove(0, new_objective_value, intra_changes, inter_changes)

    np.testing.assert_array_equal(solution_object.selection, expected_selection)
    np.testing.assert_almost_equal(solution_object.objective, expected_objective_value, decimal=5)
    assert expected_points_per_cluster == solution_object.points_per_cluster
    assert expected_selection_per_cluster == solution_object.selection_per_cluster
    assert expected_nonselection_per_cluster == solution_object.nonselection_per_cluster
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    np.testing.assert_array_almost_equal(solution_object.closest_distances_inter, expected_closest_distances_inter, decimal=5)   
    assert expected_closest_points_inter == solution_object.closest_points_inter

    assert solution_object == solution.Solution(distances, clusters, selection=expected_selection, selection_cost=selection_cost)

# Functions for calculating the groundtruth
def groundtruth_objective_value(selection, clusters, distances, selection_cost):
    # Cost for selecting items
    objective_value = np.sum(selection) * selection_cost
    # Intra cluster costs
    for idx in np.where(~selection)[0]:
        cur_min = np.inf
        for other_idx in np.where((clusters == clusters[idx]) & selection)[0]:
            cur_min = min(cur_min, distances[idx, other_idx])
        objective_value += cur_min
    # Inter cluster costs
    unique_clusters = np.unique(clusters)
    for cluster_pair in itertools.combinations(unique_clusters, 2):
        cluster_1 = np.where((clusters == cluster_pair[0]) & selection)[0]
        cluster_2 = np.where((clusters == cluster_pair[1]) & selection)[0]
        cur_max = -np.inf
        for point_pair in itertools.product(cluster_1, cluster_2):
            cur_max = max(cur_max, 1 - distances[point_pair[0], point_pair[1]])
        objective_value += cur_max
    return objective_value

def compare_tuples(t1, t2):
    t1 = t1[0]
    t2 = t2[0]
    assert len(t1) == len(t2)
    for i in range(len(t1)):
        x = np.array(t1[i])
        y = np.array(t2[i])
        np.testing.assert_array_almost_equal(x, y, decimal=5)
