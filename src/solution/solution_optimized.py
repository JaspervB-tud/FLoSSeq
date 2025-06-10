import numpy as np
from scipy.spatial.distance import squareform
import itertools
import math
from decimal import Decimal, getcontext
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, ALL_COMPLETED
import multiprocessing.shared_memory as shm
from multiprocessing import Pool, Manager
import atexit
import time, psutil, os

class Solution:
    def __init__(self, distances, clusters, selection=None, selection_cost=0.1):
        # Assert that distances and clusters have the same number of rows
        if distances.shape[0] != clusters.shape[0]:
            raise ValueError("Number of points is different between distances and clusters.")
        # Assert that distances are in [0,1] (i.e. distances are similarities or dissimilarities)
        if not np.all((distances >= 0) & (distances <= 1)):
            raise ValueError("Distances must be in the range [0, 1].")
        # If selection is provided, check if it meets criteria
        if selection is not None:
            # Assert that selection has the same number of points as clusters
            if selection.shape != clusters.shape:
                raise ValueError("Selection must have the same number of points as clusters.")
            # Assert that selection is a numpy array of booleans
            if not isinstance(selection, np.ndarray) or selection.dtype != bool:
                raise TypeError("Selection must be a numpy array of booleans.")
        else:
            selection = np.zeros(clusters.shape[0], dtype=bool)

        # Initialize object attributes
        self.selection = selection.copy()
        self.distances = squareform(distances.copy())
        self.clusters = clusters.copy()
        self.unique_clusters = np.unique(self.clusters)
        self.selection_cost = selection_cost
        self.num_points = distances.shape[0]

        # Process initial representation to optimize for comparisons speed
        self.points_per_cluster = {cluster: set(np.where(self.clusters == cluster)[0]) for cluster in self.unique_clusters} #points in every cluster
        self.selection_per_cluster = {cluster: set(np.where((self.clusters == cluster) & selection)[0]) for cluster in self.unique_clusters} #selected points in every cluster
        self.nonselection_per_cluster = {cluster: set(np.where((self.clusters == cluster) & ~selection)[0]) for cluster in self.unique_clusters} #unselected points in every cluster
        # INTRA cluster information
        self.closest_distances_intra = np.zeros(self.selection.shape[0], dtype=np.float32) #distances to closest selected point
        self.closest_points_intra = np.arange(0, self.selection.shape[0], dtype=np.int32) #indices of closest selected point
        # INTER cluster information
        self.closest_distances_inter = np.zeros((self.unique_clusters.shape[0], self.unique_clusters.shape[0]), dtype=np.float32) #distances to closest selected point
        self.closest_points_inter = {pair: (None, None) for pair in itertools.combinations(self.unique_clusters, 2)} #indices of closest selected point
        self.closest_points_inter_array = np.zeros((self.unique_clusters.shape[0], self.unique_clusters.shape[0]), dtype=np.int32) #row=from, col=to

        self.feasible = self.determine_feasibility()

        if self.feasible:
            # Set objective value
            self.objective = np.sum(self.selection) * self.selection_cost
            # INTRA cluster distances
            for idx in np.where(~self.selection)[0]:
                cur_min = np.float32(np.inf)
                cur_idx = idx
                for other_idx in np.where((self.clusters == self.clusters[idx]) & self.selection)[0]:
                    cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                    if cur_dist < cur_min:
                        cur_min = cur_dist
                        cur_idx = other_idx
                self.closest_distances_intra[idx] = cur_min
                self.closest_points_intra[idx] = cur_idx
                self.objective += cur_min
            # INTER cluster distances
            for cluster_pair in itertools.combinations(self.unique_clusters, 2):
                cluster_1 = np.where((self.clusters == cluster_pair[0]) & self.selection)[0]
                cluster_2 = np.where((self.clusters == cluster_pair[1]) & self.selection)[0]
                cur_max = -np.float32(np.inf)
                cur_pair = (None, None)
                for point_pair in itertools.product(cluster_1, cluster_2):
                    cur_dist = 1.0 - get_distance(point_pair[0], point_pair[1], self.distances, self.num_points) #WARNING: precision errors might occur here!!
                    if cur_dist > cur_max:
                        cur_max = cur_dist
                        cur_pair = point_pair
                self.closest_distances_inter[cluster_pair[0], cluster_pair[1]] = cur_max
                self.closest_distances_inter[cluster_pair[1], cluster_pair[0]] = cur_max
                self.closest_points_inter[(cluster_pair[0], cluster_pair[1])] = cur_pair  # Store the first point index
                self.closest_points_inter_array[(cluster_pair[0], cluster_pair[1])] = cur_pair[1]
                self.closest_points_inter_array[(cluster_pair[1], cluster_pair[0])] = cur_pair[0]
                self.objective += cur_max

        else:
            # If selection provided is not feasible, don't do anything (YET)
            self.objective = 0.0
            print("The solution is infeasible, objective value is set to INF and closest distances & points are not set.")

    def __eq__(self, other):
        """
        Check if two solutions are equal.
        """
        if not isinstance(other, Solution):
            return False
        # Check if distances and clusters are equal
        if not np.allclose(self.distances, other.distances, atol=1e-5) or not np.array_equal(self.clusters, other.clusters):
            return False
        # Check if selection is equal
        if self.selection is None or other.selection is None:
            return False
        # Check if selection is equal
        if self.selection.shape != other.selection.shape:
            return False
        if not np.array_equal(self.selection, other.selection):
            return False
        # Check if selection cost is equal
        if not math.isclose(self.selection_cost, other.selection_cost, rel_tol=1e-8):
            return False
        # Check if feasible is equal
        if self.feasible != other.feasible:
            return False
        if self.feasible:
            # Check if objective is equal
            if not math.isclose(self.objective, other.objective, rel_tol=1e-8):
                return False
            # Check if closest_distances_intra is equal
            if not np.allclose(self.closest_distances_intra, other.closest_distances_intra, atol=1e-5):
                return False
            # Check if closest_points_intra is equal
            if not np.array_equal(self.closest_points_intra, other.closest_points_intra):
                return False
            # Check if closest_distances_inter is equal
            if not np.allclose(self.closest_distances_inter, other.closest_distances_inter, atol=1e-5):
                return False
            # Check if closest_points_inter is equal
            if set(self.closest_points_inter.keys()) != set(other.closest_points_inter.keys()):
                return False
            for key in self.closest_points_inter:
                if self.closest_points_inter[key] != other.closest_points_inter[key]:
                    return False
        return True

    @staticmethod
    def generate_centroid_solution(distances, clusters, selection_cost=0.1):
        """
        Generates an initial solution by selecting the centroid for every cluster.

        Parameters:
        -----------
        distances: numpy.ndarray
            A 2D array where distances[i, j] represents the distance (similarity) between point i and point j.
        clusters: numpy.ndarray
            A 1D array where clusters[i] represents the cluster assignment of point i.

        Returns:
        --------
        Solution
            A solution object initialized with centroids for every cluster.
        """
        # Assert that distances and clusters have the same number of rows
        if distances.shape[0] != clusters.shape[0]:
            raise ValueError("Number of points is different between distances and clusters.")
        # Assert that distances are in [0,1] (i.e. distances are similarities or dissimilarities)
        if not np.all((distances >= 0) & (distances <= 1)):
            raise ValueError("Distances must be in the range [0, 1].")
        
        unique_clusters = np.unique(clusters)
        selection = np.zeros(clusters.shape[0], dtype=bool)

        for cluster in unique_clusters:
            cluster_points = np.where(clusters == cluster)[0]
            cluster_distances = distances[np.ix_(cluster_points, cluster_points)]
            centroid = np.argmin(np.sum(cluster_distances, axis=1))
            selection[cluster_points[centroid]] = True

        return Solution(distances, clusters, selection, selection_cost)
    
    @staticmethod
    def generate_random_solution(distances, clusters, selection_cost=0.1, max_fraction=0.1, seed=None):
        """
        Generates a random initial solution with up to X% of the points selected,
        ensuring at least one point per cluster is selected.

        Parameters:
        -----------
        distances: numpy.ndarray
            A 2D array where distances[i, j] represents the distance (similarity) between point i and point j.
        clusters: numpy.ndarray
            A 1D array where clusters[i] represents the cluster assignment of point i.
        selection_cost: float
            The cost associated with selecting a point.
        max_fraction: float
            The maximum fraction of points to select (0-1].
        seed: int, optional
            Random seed for reproducibility. Default is None (no seed).

        Returns:
        --------
        Solution
            A randomly initialized solution object.
        """
        if distances.shape[0] != clusters.shape[0]:
            raise ValueError("Number of points is different between distances and clusters.")
        if not np.all((distances >= 0) & (distances <= 1)):
            raise ValueError("Distances must be in the range [0, 1].")
        if not (0 < max_fraction <= 1):
            raise ValueError("max_fraction must be between 0 (exclusive) and 1 (inclusive).")

        unique_clusters = np.unique(clusters)
        selection = np.zeros(clusters.shape[0], dtype=bool)

        if type(seed) == int:
            np.random.seed(seed)

        # Ensure at least one point per cluster is selected
        for cluster in unique_clusters:
            cluster_points = np.where(clusters == cluster)[0]
            selected_point = np.random.choice(cluster_points)
            selection[selected_point] = True

        # Randomly select additional points up to the max_fraction limit
        num_points = clusters.shape[0]
        max_selected_points = int(max_fraction * num_points)
        remaining_points = np.where(~selection)[0]
        num_additional_points = max(0, max_selected_points - np.sum(selection))
        additional_points = np.random.choice(remaining_points, size=num_additional_points, replace=False)
        selection[additional_points] = True

        return Solution(distances, clusters, selection, selection_cost)

    def evaluate_add(self, idx_to_add, local_search=False):
        """
        Evaluates whether the proposed addition improves the current solution.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        if self.selection[idx_to_add]:
            raise ValueError("The point to add must not be selected.")
        cluster = self.clusters[idx_to_add]
        candidate_objective = self.objective + self.selection_cost # cost for adding the point
        # Calculate intra cluster distances for cluster of new point
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for idx in self.nonselection_per_cluster[cluster]:
            cur_dist = get_distance(idx, idx_to_add, self.distances, self.num_points) # distance to current point (idx)
            if cur_dist < self.closest_distances_intra[idx]:
                candidate_objective += cur_dist - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_dist))

        # NOTE: Inter-cluster distances can only increase when adding a point, so when doing local search we can exit here if objective is worse
        if candidate_objective > self.objective and np.abs(self.objective - candidate_objective) > 1e-5 and local_search:
            return np.inf, None, None

        # Inter cluster distances for other clusters
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                cur_max = self.closest_distances_inter[cluster, other_cluster]
                cur_idx = -1
                for idx in self.selection_per_cluster[other_cluster]:
                    cur_similarity = 1 - get_distance(idx, idx_to_add, self.distances, self.num_points) #this is the similarity, if it is more similar then change solution
                    if cur_similarity > cur_max:
                        cur_max = cur_similarity
                        cur_idx = idx
                if cur_idx > -1:
                    candidate_objective += cur_max - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_max, cur_idx))

        return candidate_objective, add_within_cluster, add_for_other_clusters
    
    def accept_add(self, idx_to_add, candidate_objective, add_within_cluster, add_for_other_clusters):
        """
        Accepts the addition of a point to the solution.
        Note that this assumes that the initial solution
        was feasible.
        ------------------------------------------------
        PARAMETERS:
        idx_to_add: int
            The index of the point to be added.
        candidate_objective: float
            The objective value of the solution after the addition.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, new_distance, index_in_other_cluster)]
        """
        cluster = self.clusters[idx_to_add]
        # Update selected points
        self.selection[idx_to_add] = True
        self.selection_per_cluster[cluster].add(idx_to_add)
        self.nonselection_per_cluster[cluster].remove(idx_to_add) #explicitly remove instead of discard since it should throw an error if not selected
        # Update intra cluster distances (add_within_cluster)
        for idx, dist in add_within_cluster:
            self.closest_distances_intra[idx] = dist
            self.closest_points_intra[idx] = idx_to_add
        # Update inter cluster distances (add_for_other_clusters)
        for other_cluster, dist, idx in add_for_other_clusters:
            self.closest_distances_inter[cluster, other_cluster] = dist
            self.closest_distances_inter[other_cluster, cluster] = dist
            if other_cluster < cluster:
                self.closest_points_inter[(other_cluster, cluster)] = (idx, idx_to_add)
            else:
                self.closest_points_inter[(cluster, other_cluster)] = (idx_to_add, idx)
        # Update objective value
        self.objective = candidate_objective
         
    def evaluate_swap(self, idx_to_add, idx_to_remove):
        """
        Evaluates whether the proposed swap improves the current solution.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        if self.selection[idx_to_add] or not self.selection[idx_to_remove]:
            raise ValueError("The point to add must not be selected and the point to remove must be selected.")
        cluster = self.clusters[idx_to_add]
        if cluster != self.clusters[idx_to_remove]:
            raise ValueError("The points to swap must be in the same cluster.")
        candidate_objective = self.objective
        # Generate pool of alternative points to compare to
        new_selection = set(self.selection_per_cluster[cluster])
        new_selection.add(idx_to_add)
        new_selection.remove(idx_to_remove)
        new_nonselection = set(self.nonselection_per_cluster[cluster])
        new_nonselection.add(idx_to_remove)
        # Calculate intra cluster distances for cluster of new point
        #   - check if removed point was closest selected point for any of the unselected points -> if so, replace with new point
        #   - check if added point is closest selected point for any of the unselected points -> if so, replace
        add_within_cluster = []
        for idx in new_nonselection:
            cur_closest_distance = self.closest_distances_intra[idx]
            cur_closest_point = self.closest_points_intra[idx]
            if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                cur_closest_distance = np.inf
                for other_idx in new_selection:
                    cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                    if cur_dist < cur_closest_distance:
                        cur_closest_distance = cur_dist
                        cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
            else: #point to be removed is not closest, check if newly added point is closer
                cur_dist = get_distance(idx, idx_to_add, self.distances, self.num_points)
                if cur_dist < cur_closest_distance:
                    candidate_objective += cur_dist - cur_closest_distance
                    add_within_cluster.append((idx, idx_to_add, cur_dist))
        # Calculate inter cluster distances for all other clusters
        add_for_other_clusters = []
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                cur_closest_similarity = self.closest_distances_inter[cluster, other_cluster]
                if other_cluster < cluster:
                    cur_closest_point = self.closest_points_inter[other_cluster, cluster][1]
                else:
                    cur_closest_point = self.closest_points_inter[cluster, other_cluster][0]
                cur_closest_pair = (-1, -1) #from -> to (from perspective of "other_cluster")
                if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                    cur_closest_similarity = -np.inf
                    for idx in self.selection_per_cluster[other_cluster]:
                        for other_idx in new_selection:
                            cur_similarity = 1 - get_distance(idx, other_idx, self.distances, self.num_points)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                if other_cluster < cluster:
                                    cur_closest_pair = (idx, other_idx)
                                else:
                                    cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
                else: #point to be removed is not closest, check if newly added point is closer
                    for idx in self.selection_per_cluster[other_cluster]:
                        cur_similarity = 1 - get_distance(idx, idx_to_add, self.distances, self.num_points)
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            if other_cluster < cluster:
                                cur_closest_pair = (idx, idx_to_add)
                            else:
                                cur_closest_pair = (idx_to_add, idx)
                    if cur_closest_pair[0] > -1:
                        candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                        add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
        return candidate_objective, add_within_cluster, add_for_other_clusters

    def accept_swap(self, idx_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters):
        """
        Accepts the swap of a pair of points to the solution.
        Note that this assumes that the initial solution
        was feasible.
        -----------------------------------------------------
        PARAMETERS:
        idx_to_add: int
            The index of the point to be added.
        idx_to_remove: int
            The index of the point to be removed.
        candidate_objective: float
            The objective value of the solution after the addition.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, cur_closest_pair, new_distance)]
            Note that for cur_closest_pair, the first index is in the cluster with lowest index.
        """
        cluster = self.clusters[idx_to_add]
        # Update selected points
        self.selection[idx_to_add] = True
        self.selection[idx_to_remove] = False
        self.selection_per_cluster[cluster].add(idx_to_add)
        self.selection_per_cluster[cluster].remove(idx_to_remove)
        self.nonselection_per_cluster[cluster].add(idx_to_remove)
        self.nonselection_per_cluster[cluster].remove(idx_to_add)
        # Update intra cluster distances (add_within_cluster)
        for idx, new_closest_point, dist in add_within_cluster:
            self.closest_distances_intra[idx] = dist
            self.closest_points_intra[idx] = new_closest_point
        # Update inter cluster distances (add_for_other_clusters)
        for other_cluster, cur_closest_pair, dist in add_for_other_clusters:
            self.closest_distances_inter[cluster, other_cluster] = dist
            self.closest_distances_inter[other_cluster, cluster] = dist
            if other_cluster < cluster:
                self.closest_points_inter[(other_cluster, cluster)] = cur_closest_pair
            else:
                self.closest_points_inter[(cluster, other_cluster)] = cur_closest_pair
        # Update objective value
        self.objective = candidate_objective

    def evaluate_doubleswap(self, idxs_to_add, idx_to_remove):
        """
        Evaluates whether the proposed double swap improves the current solution.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        idx_to_add1, idx_to_add2 = idxs_to_add
        if self.selection[idx_to_add1] or self.selection[idx_to_add2] or not self.selection[idx_to_remove]:
            raise ValueError("The point(s) to add must not be selected and the point to remove must be selected.")
        cluster = self.clusters[idx_to_add1]
        if cluster != self.clusters[idx_to_remove] or cluster != self.clusters[idx_to_add2]:
            raise ValueError("All points must be in the same cluster.")
        candidate_objective = self.objective + self.selection_cost
        # Generate pool of alternative points to compare to
        new_selection = set(self.selection_per_cluster[cluster])
        new_selection.add(idx_to_add1)
        new_selection.add(idx_to_add2)
        new_selection.remove(idx_to_remove)
        new_nonselection = set(self.nonselection_per_cluster[cluster])
        new_nonselection.add(idx_to_remove)
        # Calculate intra cluster distances for cluster of new point
        #   - check if removed point was closest selected point for any of the unselected points -> if so, replace with new point
        #   - check if added point is closest selected point for any of the unselected points -> if so, replace
        add_within_cluster = []
        for idx in new_nonselection:
            cur_closest_distance = self.closest_distances_intra[idx]
            cur_closest_point = self.closest_points_intra[idx]
            if cur_closest_point == idx_to_remove:
                cur_closest_distance = np.inf
                for other_idx in new_selection:
                    cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                    if cur_dist < cur_closest_distance:
                        cur_closest_distance = cur_dist
                        cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
            else: #point to be removed is not closest, check if one of newly added points is closer
                cur_dist1 = get_distance(idx, idx_to_add1, self.distances, self.num_points)
                cur_dist2 = get_distance(idx, idx_to_add2, self.distances, self.num_points)
                cur_dist, idx_to_add = min((cur_dist1, idx_to_add1), (cur_dist2, idx_to_add2), key = lambda x: x[0])
                if cur_dist < cur_closest_distance:
                    candidate_objective += cur_dist - cur_closest_distance
                    add_within_cluster.append((idx, idx_to_add, cur_dist))
        # Calculate inter cluster distances for all other clusters
        #   - Check if removed point was closest selected point for any of the other clusters -> if so replace with another point (looping over all selected points in cluster)
        #   - Otherwise, check if added points are closer to any of the selected points per cluster
        add_for_other_clusters = []
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                cur_closest_similarity = self.closest_distances_inter[cluster, other_cluster]
                if other_cluster < cluster:
                    cur_closest_point = self.closest_points_inter[other_cluster, cluster][1]
                else:
                    cur_closest_point = self.closest_points_inter[cluster, other_cluster][0]
                cur_closest_pair = (-1, -1) #from -> to (from perspective of "other_cluster")
                if cur_closest_point == idx_to_remove:
                    cur_closest_similarity = -np.inf
                    for idx in self.selection_per_cluster[other_cluster]:
                        for other_idx in new_selection:
                            cur_similarity = 1 - get_distance(idx, other_idx, self.distances, self.num_points)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                if other_cluster < cluster:
                                    cur_closest_pair = (idx, other_idx)
                                else:
                                    cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
                else: #point to be removed is not closest, check if newly added points are closer
                    for idx in self.selection_per_cluster[other_cluster]:
                        cur_similarity1 = 1 - get_distance(idx, idx_to_add1, self.distances, self.num_points)
                        cur_similarity2 = 1 - get_distance(idx, idx_to_add2, self.distances, self.num_points)
                        cur_similarity, idx_to_add = max((cur_similarity1, idx_to_add1), (cur_similarity2, idx_to_add2), key = lambda x: x[0])
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            if other_cluster < cluster:
                                cur_closest_pair = (idx, idx_to_add)
                            else:
                                cur_closest_pair = (idx_to_add, idx)
                    if cur_closest_pair[0] > -1:
                        candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                        add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))

        return candidate_objective, add_within_cluster, add_for_other_clusters

    def accept_doubleswap(self, idxs_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters):
        """
        Accepts the double swap of a pair of points to the solution.
        Note that this assumes that the initial solution
        was feasible.
        -----------------------------------------------------
        PARAMETERS:
        idxs_to_add: tuple of ints
            The indices of the points to be added.
        idx_to_remove: int
            The index of the point to be removed.
        candidate_objective: float
            The objective value of the solution after the addition.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, cur_closest_pair, new_distance)]
            Note that for cur_closest_pair, the first index is in the cluster with lowest index.
        """
        idx_to_add1, idx_to_add2 = idxs_to_add
        cluster = self.clusters[idx_to_add1]
        # Update selected points
        self.selection[idx_to_add1] = True
        self.selection[idx_to_add2] = True
        self.selection[idx_to_remove] = False
        self.selection_per_cluster[cluster].add(idx_to_add1)
        self.selection_per_cluster[cluster].add(idx_to_add2)
        self.selection_per_cluster[cluster].remove(idx_to_remove)
        self.nonselection_per_cluster[cluster].add(idx_to_remove)
        self.nonselection_per_cluster[cluster].remove(idx_to_add1)
        self.nonselection_per_cluster[cluster].remove(idx_to_add2)
        # Update intra cluster distances (add_within_cluster)
        for idx, new_closest_point, dist in add_within_cluster:
            self.closest_distances_intra[idx] = dist
            self.closest_points_intra[idx] = new_closest_point
        # Update inter cluster distances (add_for_other_clusters)
        for other_cluster, cur_closest_pair, dist in add_for_other_clusters:
            self.closest_distances_inter[cluster, other_cluster] = dist
            self.closest_distances_inter[other_cluster, cluster] = dist
            if other_cluster < cluster:
                self.closest_points_inter[(other_cluster, cluster)] = cur_closest_pair
            else:
                self.closest_points_inter[(cluster, other_cluster)] = cur_closest_pair
        # Update objective value
        self.objective = candidate_objective

    def evaluate_remove(self, idx_to_remove, local_search=False):
        """
        Evaluates whether the proposed removal improves the current solution.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        if not self.selection[idx_to_remove]:
            raise ValueError("The point to remove must be selected.")
        cluster = self.clusters[idx_to_remove]
        if len(self.selection_per_cluster[cluster]) == 1:
            raise ValueError("The point to remove is the only selected point in its cluster.")
        candidate_objective = self.objective - self.selection_cost
        # Generate pool of alternative points to compare to
        new_selection = set(self.selection_per_cluster[cluster])
        new_selection.discard(idx_to_remove)
        new_nonselection = set(self.nonselection_per_cluster[cluster])
        new_nonselection.add(idx_to_remove)

        # Calculate inter cluster distances for all other clusters (start with inter here for local_search optimization)
        #  - Check if removed point was closest selected point for any of the other clusters -> if so replace with another point (looping over all selected points in cluster)
        add_for_other_clusters = []
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                cur_closest_similarity = self.closest_distances_inter[cluster, other_cluster]
                if other_cluster < cluster:
                    cur_closest_point = self.closest_points_inter[other_cluster, cluster][1]
                else:
                    cur_closest_point = self.closest_points_inter[cluster, other_cluster][0]
                cur_closest_pair = (-1, -1) #from - to (considered from perspective of "other_cluster")
                if cur_closest_point == idx_to_remove:
                    cur_closest_similarity = -np.inf
                    for idx in self.selection_per_cluster[other_cluster]:
                        for other_idx in new_selection:
                            cur_similarity = 1 - get_distance(idx, other_idx, self.distances, self.num_points)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                if other_cluster < cluster:
                                    cur_closest_pair = (idx, other_idx)
                                else:
                                    cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
        
        # NOTE: Intra-cluster distances can only increase when removing a point, so when doing local search we can exit here if objective is worse
        if candidate_objective > self.objective and np.abs(self.objective - candidate_objective) > 1e-5 and local_search:
            return np.inf, None, None

        # Calculate intra cluster distances for cluster of removed point
        #   - Check if removed point was closest selected point for any of the unselected points -> if so, replace with new point
        add_within_cluster = []
        for idx in new_nonselection:
            cur_closest_distance = self.closest_distances_intra[idx]
            cur_closest_point = self.closest_points_intra[idx]
            if cur_closest_point == idx_to_remove:
                cur_closest_distance = np.inf
                for other_idx in new_selection:
                    if other_idx != idx:
                        cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                        if cur_dist < cur_closest_distance:
                            cur_closest_distance = cur_dist
                            cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
        
        
        return candidate_objective, add_within_cluster, add_for_other_clusters

    def accept_remove(self, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters):
        """
        Accepts the removal of a point from the solution.
        Note that this assumes that the initial solution
        and resulting solution are feasible.
        -----------------------------------------------------
        PARAMETERS:
        idx_to_remove: int
            The index of the point to be removed.
        candidate_objective: float
            The objective value of the solution after the removal.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the removed point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, cur_closest_pair, new_distance)]
            Note that for cur_closest_pair, the first index is in the cluster with lowest index.
        """
        cluster = self.clusters[idx_to_remove]
        # Update selected points
        self.selection[idx_to_remove] = False
        self.selection_per_cluster[cluster].remove(idx_to_remove)
        self.nonselection_per_cluster[cluster].add(idx_to_remove)
        # Update intra cluster distances (add_within_cluster)
        for idx, new_closest_point, dist in add_within_cluster:
            self.closest_distances_intra[idx] = dist
            self.closest_points_intra[idx] = new_closest_point
        # Update inter cluster distances (add_for_other_clusters)
        for other_cluster, cur_closest_pair, dist in add_for_other_clusters:
            self.closest_distances_inter[cluster, other_cluster] = dist
            self.closest_distances_inter[other_cluster, cluster] = dist
            if other_cluster < cluster:
                self.closest_points_inter[(other_cluster, cluster)] = cur_closest_pair
            else:
                self.closest_points_inter[(cluster, other_cluster)] = cur_closest_pair
        # Update objective value
        self.objective = candidate_objective

    def determine_feasibility(self):
        uncovered_clusters = set(self.unique_clusters)
        for point in np.where(self.selection)[0]:
            uncovered_clusters.discard(self.clusters[point])
        return len(uncovered_clusters) == 0

    def local_search(self, max_iterations=1000, best_swap=False):
        """
        Perform a local search to find a (local) optimal solution.
        
        Parameters:
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        best_swap: bool
            If True, only the best swap will be performed in each iteration, instead of the first swap that improves the solution.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")
        
        iteration = 0
        objectives = [(self.objective, "start")]
        selections = [self.selection.copy()]

        solution_changed = False
        while iteration < max_iterations:
            solution_changed = False
            # Try adding a point
            for idx_to_add in self.generate_indices_add():
                candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add)
                if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                    self.accept_add(idx_to_add, candidate_objective, add_within_cluster, add_for_other_clusters)
                    solution_changed = True
                    objectives.append((self.objective, f"added {idx_to_add} from cluster {self.clusters[idx_to_add]}"))
                    selections.append(self.selection.copy())
                    break
            # Try swapping a selected point and unselected point
            if not solution_changed:
                if best_swap:
                    best_swap_candidate = (self.objective, None, None, None, None, False)
                    for idx_to_add, idx_to_remove in self.generate_indices_swap():
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                        if candidate_objective < best_swap_candidate[0] and np.abs(candidate_objective - best_swap_candidate[0]) > 1e-8:
                            best_swap_candidate = (candidate_objective, idx_to_add, idx_to_remove, add_within_cluster, add_for_other_clusters, True)
                    if best_swap_candidate[5]:
                        self.accept_swap(best_swap_candidate[1], best_swap_candidate[2], best_swap_candidate[0], best_swap_candidate[3], best_swap_candidate[4])
                        solution_changed = True
                        objectives.append((self.objective, f"swapped {best_swap_candidate[2]} for {best_swap_candidate[1]} in cluster {self.clusters[best_swap_candidate[1]]}"))
                        selections.append(self.selection.copy())
                else:
                    for idx_to_add, idx_to_remove in self.generate_indices_swap():
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                            self.accept_swap(idx_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                            solution_changed = True
                            objectives.append((self.objective, f"swapped {idx_to_remove} for {idx_to_add} in cluster {self.clusters[idx_to_add]}"))
                            selections.append(self.selection.copy())
                            break
            # Try swapping two selected points with one unselected point
            if not solution_changed:
                for (idx_to_add1, idx_to_add2), idx_to_remove in self.generate_indices_doubleswap():
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        objectives.append((self.objective, f"doubleswapped {idx_to_remove} for {idx_to_add1} and {idx_to_add2} in cluster {self.clusters[idx_to_add1]}"))
                        selections.append(self.selection.copy())
                        break
            # Try removing a point
            if not solution_changed:
                for idx_to_remove in self.generate_indices_remove():
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_remove(idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        objectives.append((self.objective, f"removed {idx_to_remove} from cluster {self.clusters[idx_to_remove]}"))
                        selections.append(self.selection.copy())
                        break
            if not solution_changed:
                break
            else:
                iteration += 1
                if iteration % 200 == 0:
                    print(f"Iteration {iteration}: Objective = {self.objective}")

        return objectives, selections

    def local_search_random(self, max_iterations=1000, seed=None):
        """
        Perform a local search to find a (local) optimal solution.
        NOTE: This version picks a random move to make rather than structurally exhausting all options.
        
        Parameters:
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")
        
        iteration = 0
        objectives = [(self.objective, "start")]
        selections = [self.selection.copy()]
        time_per_iteration = []

        if type(seed) is int:
            np.random.seed(seed)

        solution_changed = False
        while iteration < max_iterations:
            solution_changed = False
            cur_time = time.time()
            for move_type, move_content in self.generate_random_moves(seed=seed):
                if move_type == "add":
                    idx_to_add = move_content
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add, local_search=True)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_add(idx_to_add, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        #objectives.append((self.objective, f"added {idx_to_add} from cluster {self.clusters[idx_to_add]}"))
                        #selections.append(self.selection.copy())
                        break
                elif move_type == "swap":
                    idx_to_add, idx_to_remove = move_content
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_swap(idx_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        #objectives.append((self.objective, f"swapped {idx_to_remove} for {idx_to_add} in cluster {self.clusters[idx_to_add]}"))
                        #selections.append(self.selection.copy())
                        break
                elif move_type == "doubleswap":
                    (idx_to_add1, idx_to_add2), idx_to_remove = move_content
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        #objectives.append((self.objective, f"doubleswapped {idx_to_remove} for {idx_to_add1} and {idx_to_add2} in cluster {self.clusters[idx_to_add1]}"))
                        #selections.append(self.selection.copy())
                        break
                elif move_type == "remove":
                    idx_to_remove = move_content
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove, local_search=True)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_remove(idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        #objectives.append((self.objective, f"removed {idx_to_remove} from cluster {self.clusters[idx_to_remove]}"))
                        #selections.append(self.selection.copy())
                        break
            time_per_iteration.append(time.time() - cur_time)
            if not solution_changed:
                break
            else:
                #print(move_type, move_content, candidate_objective)
                iteration += 1
                if iteration % 50 == 0:
                    print(f"Iteration {iteration}: Objective = {self.objective}", flush=True)
        #print(time_per_iteration)
        return objectives, selections, time_per_iteration

    def local_search_removefirst(self, max_iterations=1000, best_swap=False):
        """
        Perform a local search to find a (local) optimal solution.
        
        Parameters:
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        best_swap: bool
            If True, only the best swap will be performed in each iteration, instead of the first swap that improves the solution.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")
        
        iteration = 0
        objectives = [(self.objective, "start")]
        selections = [self.selection.copy()]

        solution_changed = False
        while iteration < max_iterations:
            solution_changed = False
            # Try removing a point
            for idx_to_remove in self.generate_indices_remove():
                candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove)
                if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                    self.accept_remove(idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                    solution_changed = True
                    objectives.append((self.objective, f"removed {idx_to_remove} from cluster {self.clusters[idx_to_remove]}"))
                    selections.append(self.selection.copy())
                    break
            # Try adding a point
            if not solution_changed:
                for idx_to_add in self.generate_indices_add():
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_add(idx_to_add, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        objectives.append((self.objective, f"added {idx_to_add} from cluster {self.clusters[idx_to_add]}"))
                        selections.append(self.selection.copy())
                        break
            # Try swapping a selected point and unselected point
            if not solution_changed:
                if best_swap:
                    best_swap_candidate = (self.objective, None, None, None, None, False)
                    for idx_to_add, idx_to_remove in self.generate_indices_swap():
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                        if candidate_objective < best_swap_candidate[0] and np.abs(candidate_objective - best_swap_candidate[0]) > 1e-8:
                            best_swap_candidate = (candidate_objective, idx_to_add, idx_to_remove, add_within_cluster, add_for_other_clusters, True)
                    if best_swap_candidate[5]:
                        self.accept_swap(best_swap_candidate[1], best_swap_candidate[2], best_swap_candidate[0], best_swap_candidate[3], best_swap_candidate[4])
                        solution_changed = True
                        objectives.append((self.objective, f"swapped {best_swap_candidate[2]} for {best_swap_candidate[1]} in cluster {self.clusters[best_swap_candidate[1]]}"))
                        selections.append(self.selection.copy())
                else:
                    for idx_to_add, idx_to_remove in self.generate_indices_swap():
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                            self.accept_swap(idx_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                            solution_changed = True
                            objectives.append((self.objective, f"swapped {idx_to_remove} for {idx_to_add} in cluster {self.clusters[idx_to_add]}"))
                            selections.append(self.selection.copy())
                            break
            # Try swapping two selected points with one unselected point
            if not solution_changed:
                for (idx_to_add1, idx_to_add2), idx_to_remove in self.generate_indices_doubleswap():
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        objectives.append((self.objective, f"doubleswapped {idx_to_remove} for {idx_to_add1} and {idx_to_add2} in cluster {self.clusters[idx_to_add1]}"))
                        selections.append(self.selection.copy())
                        break
            
            if not solution_changed:
                break
            else:
                iteration += 1
                if iteration % 200 == 0:
                    print(f"Iteration {iteration}: Objective = {self.objective}")

        return objectives, selections

    def local_search_parallel(self, max_iterations=1000, best_swap=False, num_cores=None):
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")
        
        iteration = 0
        objectives = [(self.objective, "start")]
        solution_changed = False

        # These should be parameters to the function, but are hardcoded for now for simplicity
        batch_size = 1000
        max_batches = 20
        try:
            # Copy distance matrix to shared memory
            distances_shm = shm.SharedMemory(create=True, size=self.distances.nbytes)
            shared_distances = np.ndarray(self.distances.shape, dtype=self.distances.dtype, buffer=distances_shm.buf)
            np.copyto(shared_distances, self.distances)
            # INTRA
            # Copy closest_distances_intra to shared memory
            closest_distances_intra_shm = shm.SharedMemory(create=True, size=self.closest_distances_intra.nbytes)
            shared_closest_distances_intra = np.ndarray(self.closest_distances_intra.shape, dtype=self.closest_distances_intra.dtype, buffer=closest_distances_intra_shm.buf)
            
            # Copy closest_points_intra to shared memory
            closest_points_intra_shm = shm.SharedMemory(create=True, size=self.closest_points_intra.nbytes)
            shared_closest_points_intra = np.ndarray(self.closest_points_intra.shape, dtype=self.closest_points_intra.dtype, buffer=closest_points_intra_shm.buf)
            
            # INTER
            # Copy closest_distances_inter to shared memory
            closest_distances_inter_shm = shm.SharedMemory(create=True, size=self.closest_distances_inter.nbytes)
            shared_closest_distances_inter = np.ndarray(self.closest_distances_inter.shape, dtype=self.closest_distances_inter.dtype, buffer=closest_distances_inter_shm.buf)
            

            with Manager() as manager:
                event = manager.Event() #this is used to signal when tasks should be stopped
                results = manager.list()
                with Pool(
                    processes=num_cores,
                    initializer=init_worker,
                    initargs=(
                        distances_shm.name, shared_distances.shape,
                        closest_distances_intra_shm.name, shared_closest_distances_intra.shape,
                        closest_points_intra_shm.name, shared_closest_points_intra.shape,
                        closest_distances_inter_shm.name, shared_closest_distances_inter.shape,
                        self.unique_clusters, self.selection_cost, self.num_points),
                    ) as pool:
                    # Construct outer while loop that iterates until local optimum is found, or max_iterations is reached
                    start = time.time()
                    while iteration < max_iterations:
                        # Update closest distances and points in shared memory
                        np.copyto(shared_closest_distances_intra, self.closest_distances_intra)
                        np.copyto(shared_closest_points_intra, self.closest_points_intra)
                        np.copyto(shared_closest_distances_inter, self.closest_distances_inter)
                        
                        print(f"Iteration {iteration+1}, objective={self.objective:.5f}", flush=True)

                        move_generator = self.generate_random_moves(seed=1234)
                        
                        batch_id = 0
                        # This loop makes batches of batches of moves each to be processed in parallel.
                        while True:
                            if event.is_set():
                                break
                            batches = [] #batch of batches
                            for _ in range(max_batches): #create batches of moves
                                batch = [] #batch of moves
                                try:
                                    for _ in range(batch_size):
                                        move_type, move_content = next(move_generator)
                                        batch.append((move_type, move_content))
                                except StopIteration: #if no more moves available, break
                                    if len(batch) > 0:
                                        batches.append(batch)
                                    break
                                if len(batch) > 0:
                                    batches.append(batch)
                            
                            # Process current collection of batches in parallel
                            if len(batches) > 0:
                                #print("Current batch:", batch_id, len(batches), batches, flush=True)
                                batch_id += 1
                                batch_results = []
                                for b in batches:
                                    if event.is_set():
                                        break
                                    res = pool.apply_async(
                                        test,
                                        (b,event),
                                    )
                                    batch_results.append(res)
                                
                                for result in batch_results:
                                    result.wait()
                            else:
                                break
                        iteration += 1
                    print("Total time taken:", time.time() - start, "seconds", flush=True)


        finally:
            distances_shm.close()
            distances_shm.unlink()

            closest_distances_intra_shm.close()
            closest_distances_intra_shm.unlink()

            closest_points_intra_shm.close()
            closest_points_intra_shm.unlink()

            closest_distances_inter_shm.close()
            closest_distances_inter_shm.unlink()

    def local_search_parallel_exp(self, max_iterations=1000, best_swap=False, num_cores=None, seed=1234, hybrid=False, batch_size=100, max_batches=10):
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")
        
        iteration = 0
        objectives = [(self.objective, "start")]
        time_per_iteration = []
        switch_iteration = -1
        solution_changed = False

        try:
            # Copy distance matrix to shared memory
            distances_shm = shm.SharedMemory(create=True, size=self.distances.nbytes)
            shared_distances = np.ndarray(self.distances.shape, dtype=self.distances.dtype, buffer=distances_shm.buf)
            np.copyto(shared_distances, self.distances)
            # Copy cluster assignment to shared memory
            clusters_shm = shm.SharedMemory(create=True, size=self.clusters.nbytes)
            shared_clusters = np.ndarray(self.clusters.shape, dtype=self.clusters.dtype, buffer=clusters_shm.buf)
            np.copyto(shared_clusters, self.clusters)
            # INTRA
            # Copy closest_distances_intra to shared memory
            closest_distances_intra_shm = shm.SharedMemory(create=True, size=self.closest_distances_intra.nbytes)
            shared_closest_distances_intra = np.ndarray(self.closest_distances_intra.shape, dtype=self.closest_distances_intra.dtype, buffer=closest_distances_intra_shm.buf)
            # Copy closest_points_intra to shared memory
            closest_points_intra_shm = shm.SharedMemory(create=True, size=self.closest_points_intra.nbytes)
            shared_closest_points_intra = np.ndarray(self.closest_points_intra.shape, dtype=self.closest_points_intra.dtype, buffer=closest_points_intra_shm.buf)
            # INTER
            # Copy closest_distances_inter to shared memory
            closest_distances_inter_shm = shm.SharedMemory(create=True, size=self.closest_distances_inter.nbytes)
            shared_closest_distances_inter = np.ndarray(self.closest_distances_inter.shape, dtype=self.closest_distances_inter.dtype, buffer=closest_distances_inter_shm.buf)
            
            RUNTIME_SWITCH = 2 # seconds after which we switch to multiprocessing
            NUM_SWITCH = 2 # number of times runtime has to exceed RUNTIME_SWITCH before we switch to multiprocessing

            # In hybrid mode we switch to multiprocessing once an iteration takes longer than 1 minute since there is no speed-up otherwise.
            if hybrid:
                last_run_time = 0
                times_exceeded = 0
            else:
                last_run_time = np.inf
                times_exceeded = np.inf
            

            total_time = time.time()
            with Manager() as manager:
                event = manager.Event() #this is used to signal when tasks should be stopped
                results = manager.list() #this is used to store an improvement if it exists

                with Pool(
                    processes=num_cores,
                    initializer=init_worker,
                    initargs=(
                        distances_shm.name, shared_distances.shape,
                        clusters_shm.name, shared_clusters.shape,
                        closest_distances_intra_shm.name, shared_closest_distances_intra.shape,
                        closest_points_intra_shm.name, shared_closest_points_intra.shape,
                        closest_distances_inter_shm.name, shared_closest_distances_inter.shape,
                        self.unique_clusters, self.selection_cost, self.num_points),
                    ) as pool:
                    
                    # Construct outer while loop that iterates until local optimum is found, or max_iterations is reached
                    while iteration < max_iterations:
                        solution_changed = False
                        # This allows for hybrid execution where fast iterations are done sequentially
                        if times_exceeded > NUM_SWITCH: # if enough runs took long enough, switch to multiprocessing
                            # Update closest distances and points in shared memory
                            np.copyto(shared_closest_distances_intra, self.closest_distances_intra)
                            np.copyto(shared_closest_points_intra, self.closest_points_intra)
                            np.copyto(shared_closest_distances_inter, self.closest_distances_inter)

                            total_improves = 0
                            total_moves = 0

                            # Set up for current iteration
                            move_generator = self.generate_random_moves(seed=seed)
                            event.clear() #unset event
                            results = [] #reset results for this iteration

                            cur_iteration_time = time.time()
                            # This loop makes batches of batches of moves each to be processed in parallel.
                            while True:
                                batches = [] #list of batches
                                for _ in range(max_batches): #fill the list with up to max_batches batches
                                    batch = [] #batch of moves
                                    try:
                                        for _ in range(batch_size):
                                            move_type, move_content = next(move_generator)
                                            batch.append((move_type, move_content))
                                    except StopIteration: #if no more moves available, break
                                        if len(batch) > 0:
                                            batches.append(batch)
                                        break
                                    if len(batch) > 0:
                                        batches.append(batch)
                                # At this point batches is a list with a list of moves for every entry

                                # Process current collection of batches in parallel
                                if len(batches) > 0: #if there are tasks to process
                                    batch_results = []
                                    for b in batches:
                                        if event.is_set():
                                            break
                                        res = pool.apply_async(
                                            process_batch,
                                            args=(b, event, self.closest_points_inter, self.selection_per_cluster, self.nonselection_per_cluster, self.objective),
                                            callback = lambda result: process_batch_result(result, results)
                                        )
                                        batch_results.append(res)
                                    
                                    for result in batch_results:
                                        result.wait()

                                    if len(results) > 0:
                                        solution_changed = True
                                        break
                                else: #no more tasks to process
                                    break
                            time_per_iteration.append(time.time() - cur_iteration_time)
                            if solution_changed: # if solution changed, update solution
                                move_type = results[0][0]
                                move_content = results[0][1]
                                candidate_objective = results[0][2]
                                add_within_cluster = results[0][3]
                                add_for_other_clusters = results[0][4]
                                if move_type == "add":
                                    self.accept_add(move_content, candidate_objective, add_within_cluster, add_for_other_clusters)
                                elif move_type == "swap":
                                    idx_to_add, idx_to_remove = move_content
                                    self.accept_swap(idx_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                                elif move_type == "doubleswap":
                                    (idx_to_add1, idx_to_add2), idx_to_remove = move_content
                                    self.accept_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                                elif move_type == "remove": 
                                    idx_to_remove = move_content
                                    self.accept_remove(idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                                #print(len(results), results[0][0], results[0][1], results[0][2], flush=True)
                                iteration += 1
                                if iteration % 50 == 0:
                                    print(f"Iteration {iteration}: Objective = {self.objective}", flush=True)
                            else: #stop when local optimum is found
                                break
                        else:
                            last_run_time = 0
                            start = time.time()
                            for move_type, move_content in self.generate_random_moves(seed=seed):
                                if move_type == "add":
                                    idx_to_add = move_content
                                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add, local_search=True)
                                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-5:
                                        solution_changed=True
                                        break
                                elif move_type == "swap":
                                    idx_to_add, idx_to_remove = move_content
                                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-5:
                                        solution_changed=True
                                        break
                                elif move_type == "doubleswap":
                                    (idx_to_add1, idx_to_add2), idx_to_remove = move_content
                                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove)
                                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-5:
                                        solution_changed=True
                                        break
                                elif move_type == "remove":
                                    idx_to_remove = move_content
                                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove, local_search=True)
                                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-5:
                                        solution_changed=True
                                        break
                            if hybrid:
                                last_run_time = time.time() - start
                                time_per_iteration.append(last_run_time)
                                if last_run_time > RUNTIME_SWITCH:
                                    times_exceeded += 1
                                if times_exceeded > NUM_SWITCH:
                                    print(f"Switching to multiprocessing after {iteration} iterations", flush=True)
                                    switch_iteration = iteration
                            if solution_changed:
                                if move_type == "add":
                                    self.accept_add(move_content, candidate_objective, add_within_cluster, add_for_other_clusters)
                                elif move_type == "swap":
                                    idx_to_add, idx_to_remove = move_content
                                    self.accept_swap(idx_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                                elif move_type == "doubleswap":
                                    (idx_to_add1, idx_to_add2), idx_to_remove = move_content
                                    self.accept_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                                elif move_type == "remove": 
                                    idx_to_remove = move_content
                                    self.accept_remove(idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                                #print(move_type, move_content, candidate_objective, flush=True)
                                iteration += 1
                                if iteration % 50 == 0:
                                    print(f"Iteration {iteration}: Objective = {self.objective}", flush=True)
                            else:
                                break
            #print(time.time() - total_time)
            #print(time_per_iteration)
            return time_per_iteration, switch_iteration
        except Exception as e:
            print(f"An error occurred during local search: {e}", flush=True)
            raise e
        finally:
            distances_shm.close()
            distances_shm.unlink()

            clusters_shm.close()
            clusters_shm.unlink()

            closest_distances_intra_shm.close()
            closest_distances_intra_shm.unlink()

            closest_points_intra_shm.close()
            closest_points_intra_shm.unlink()

            closest_distances_inter_shm.close()
            closest_distances_inter_shm.unlink()

    def simulated_annealing(self, max_iterations=1000, initial_temperature=1.0, cooling_rate=0.99):
        """
        Perform simulated annealing to find a (local) optimal solution.
        
        Parameters:
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        initial_temperature: float
            The initial temperature for the annealing process.
        cooling_rate: float
            The rate at which the temperature decreases.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform simulated annealing.")
        
        iteration = 0
        temperature = initial_temperature
        objectives = [(self.objective, "start")]
        selections = [self.selection.copy()]

        solution_changed = False
        times_unchanged = 0
        while iteration < max_iterations:
            solution_changed = False
            operations = [
                ("add", self.generate_indices_add),
                ("swap", self.generate_indices_swap),
                ("doubleswap", self.generate_indices_doubleswap),
                ("remove", self.generate_indices_remove),
            ]
            np.random.shuffle(operations)  # Randomize the order of operations

            for operation, generator in operations:
                if operation == "add":
                    for idx_to_add in generator():
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add)
                        delta_objective = round(candidate_objective - self.objective, 6)
                        if delta_objective < 0 or np.random.rand() < np.exp(-delta_objective / temperature):
                            self.accept_add(idx_to_add, candidate_objective, add_within_cluster, add_for_other_clusters)
                            solution_changed = True
                            objectives.append((self.objective, f"added {idx_to_add} from cluster {self.clusters[idx_to_add]}"))
                            selections.append(self.selection.copy())
                            break

                elif operation == "swap":
                    for idx_to_add, idx_to_remove in generator():
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                        delta_objective = round(candidate_objective - self.objective, 6)
                        if delta_objective < 0 or np.random.rand() < np.exp(-delta_objective / temperature):
                            self.accept_swap(idx_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                            solution_changed = True
                            objectives.append((self.objective, f"swapped {idx_to_remove} for {idx_to_add} in cluster {self.clusters[idx_to_add]}"))
                            selections.append(self.selection.copy())
                            break

                elif operation == "doubleswap":
                    for (idx_to_add1, idx_to_add2), idx_to_remove in generator():
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove)
                        delta_objective = round(candidate_objective - self.objective, 6)
                        if delta_objective < 0 or np.random.rand() < np.exp(-delta_objective / temperature):
                            self.accept_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                            solution_changed = True
                            objectives.append((self.objective, f"doubleswapped {idx_to_remove} for {idx_to_add1} and {idx_to_add2} in cluster {self.clusters[idx_to_add1]}"))
                            selections.append(self.selection.copy())
                            break

                elif operation == "remove":
                    for idx_to_remove in generator():
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove)
                        delta_objective = round(candidate_objective - self.objective, 6)
                        if delta_objective < 0 or np.random.rand() < np.exp(-delta_objective / temperature):
                            self.accept_remove(idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                            solution_changed = True
                            objectives.append((self.objective, f"removed {idx_to_remove} from cluster {self.clusters[idx_to_remove]}"))
                            selections.append(self.selection.copy())
                            break

                if solution_changed:
                    times_unchanged = 0
                    break
                else:
                    times_unchanged += 1
                    if times_unchanged > 10:
                        return objectives, selections
                    
            # Cool down the temperature
            temperature *= cooling_rate
            iteration += 1

            if iteration % 50 == 0:
                print(f"Iteration {iteration}: Objective = {self.objective}, Temperature = {temperature}")

        return objectives, selections

    def generate_indices_add(self):
        """
        Generates a list of indices to add to the solution.
        """
        yield from np.flatnonzero(~self.selection)

    def generate_indices_swap(self):
        """
        Generates a list of indices to swap in the solution.
        """
        for cluster in self.unique_clusters:
            clusters_mask = self.clusters == cluster
            selected = np.where(clusters_mask & self.selection)[0]
            unselected = np.where(clusters_mask & ~self.selection)[0]
            for pair in itertools.product(unselected, selected):
                yield pair

    def generate_indices_doubleswap(self):
        """
        Generates a list of indices to swap in the solution.
        """
        for cluster in self.unique_clusters:
            clusters_mask = self.clusters == cluster
            selected = np.where(clusters_mask & self.selection)[0]
            unselected = np.where(clusters_mask & ~self.selection)[0]
            for idx in selected:
                for other_1, other_2 in itertools.combinations(unselected, 2):
                    yield (other_1, other_2), idx

    def generate_indices_remove(self):
        """
        Generates a list of indices to remove from the solution.
        """
        for cluster in self.unique_clusters:
            if len(self.selection_per_cluster[cluster]) > 1:
                for idx in self.selection_per_cluster[cluster]:
                    yield idx

    def generate_random_moves(self, seed=None):
        """
        Creates a generator that randomly picks from the existing generators
        (add, swap, doubleswap, remove) until all of them are empty.
        """
        generators = {
            "add": self.generate_indices_add(),
            "swap": self.generate_indices_swap(),
            "doubleswap": self.generate_indices_doubleswap(),
            "remove": self.generate_indices_remove(),
        }
        active_generators = list(generators.keys())

        if seed is not None:
            np.random.seed(seed)

        while active_generators:
            # Randomly pick an active generator
            selected_generator = np.random.choice(active_generators)
            try:
                # Yield the next value from the selected generator
                yield selected_generator, next(generators[selected_generator])
            except StopIteration:
                # Remove the generator if it is exhausted
                active_generators.remove(selected_generator)
    
"""
Here we define helper functions that can be used by the multiprocessing version of the local search.
"""
def evaluate_add_helper_shm(
        idx_to_add, 
        distances, cluster, unique_clusters, objective, selection_cost, num_points,
        selection_per_cluster, nonselection_per_cluster,
        closest_distances_intra, closest_distances_inter):
        """
        Evaluates the effect of adding a point to the solution without relying on an explicit instance
        of the Solution class.
        NOTE: This function is designed to be used with shared memory for parallel processing!
        NOTE: In the current implementation, there is no check for feasibility, so it is assumed
                that the point can be added without violating any constraints!
        Parameters:
        -----------
        idx_to_add: int
            The index of the point to be added.
        distances: tuple
            A tuple containing:
            - shared memory name for the distances matrix.
            - shape of the distance matrix.
            - dtype of the distance matrix.
        clusters: np.ndarray
            The cluster assignments for each point.
        unique_clusters: np.ndarray
            The unique clusters present in the solution.
        objective: float
            The current objective value of the solution.
        selection_cost: float
            The cost associated with selecting a point.
        num_points: int
            The total number of points in the dataset.
        selection_per_cluster: dict
            A dictionary mapping each cluster to a set of indices of selected points in that cluster.
        nonselection_per_cluster: dict
            A dictionary mapping each cluster to a set of indices of non-selected points in that cluster.
        closest_distances_intra: np.ndarray
            A 1D array containing the closest intra-cluster distances for each point.
        closest_distances_inter: np.ndarray
            An array containing the closest inter-cluster distances between clusters.
        Returns:
        --------
        candidate_objective: float
            The objective value if the point is added.
        add_within_cluster: list
            A list of tuples (index, distance) for points within the cluster of the new point that would be updated.
        add_for_other_clusters: list
            A list of tuples (other_cluster, similarity, index) for points in other clusters that would be updated.
        """
        #cluster = clusters[idx_to_add]
        candidate_objective = objective + selection_cost # cost for adding the point
        try: #encapsulate in try-except-finally to handle shared memory
            D = shm.SharedMemory(name=distances[0])
            D_array = np.ndarray(distances[1], dtype=np.float32, buffer=D.buf)

            # Calculate intra cluster distances for cluster of new point
            add_within_cluster = [] #this stores changes that have to be made if the objective improves
            for idx in nonselection_per_cluster[cluster]:
                cur_dist = get_distance(idx, idx_to_add, D_array, num_points) # distance to current point (idx)
                if cur_dist < closest_distances_intra[idx]:
                    candidate_objective += cur_dist - closest_distances_intra[idx]
                    add_within_cluster.append((idx, cur_dist))

            # Calculate inter cluster distances for all other clusters
            add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
            for other_cluster in unique_clusters:
                if other_cluster != cluster:
                    cur_max = get_distance(cluster, other_cluster, closest_distances_inter, unique_clusters.shape[0])
                    cur_idx = -1
                    for idx in selection_per_cluster[other_cluster]:
                        cur_similarity = 1.0 - get_distance(idx, idx_to_add, D_array, num_points) #this is the similarity, if it is more similar then change solution
                        if cur_similarity > cur_max:
                            cur_max = cur_similarity
                            cur_idx = idx
                    if cur_idx > -1:
                        candidate_objective += cur_max - get_distance(cluster, other_cluster, closest_distances_inter, unique_clusters.shape[0])
                        add_for_other_clusters.append((other_cluster, cur_max, cur_idx))

            return candidate_objective, add_within_cluster, add_for_other_clusters
        except Exception as e:
            print(f"Error in evaluate_add_helper_shm: {e}")
            return None, None, None
        finally:
            D.close()

def evaluate_add_mp(
        idx_to_add, objective,
        selection_per_cluster, nonselection
        ):
        """
        Evaluates the effect of adding a point to the solution without relying on an explicit instance
        of the Solution class.
        NOTE: This function is designed to be used with shared memory for parallel processing!
        NOTE: In the current implementation, there is no check for feasibility, so it is assumed
                that the point can be added without violating any constraints!
        """
        cluster = _clusters[idx_to_add]
        candidate_objective = objective + _selection_cost # cost for adding the point
        
        # Intra cluster distances for same cluster
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for idx in nonselection:
            cur_dist = get_distance(idx, idx_to_add, _distances, _num_points) # distance to current point (idx)
            if cur_dist < _closest_distances_intra[idx]:
                candidate_objective += cur_dist - _closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_dist))

        # Inter cluster distances for other clusters
        # NOTE: This can only increase the inter cluster cost, so if objective is already worse, we can skip this
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        if candidate_objective > objective:
            return -1, -1, -1 #-1, -1, -1 to signify no improvement
        for other_cluster in _unique_clusters:
            if other_cluster != cluster:
                #cur_max = get_distance(cluster, other_cluster, _closest_distances_inter, _unique_clusters.shape[0])
                cur_max = _closest_distances_inter[cluster, other_cluster]
                cur_idx = -1
                for idx in selection_per_cluster[other_cluster]:
                    cur_similarity = 1.0 - get_distance(idx, idx_to_add, _distances, _num_points)
                    if cur_similarity > cur_max:
                        cur_max = cur_similarity
                        cur_idx = idx
                if cur_idx > -1:
                    #candidate_objective += cur_max - get_distance(cluster, other_cluster, _closest_distances_inter, _unique_clusters.shape[0])
                    candidate_objective += cur_max - _closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_max, cur_idx))

        if candidate_objective < objective:
            return candidate_objective, add_within_cluster, add_for_other_clusters
        else:
            return -1, -1, -1  # -1, -1, -1 to signify no improvement

def evaluate_swap_mp(
        idxs_to_add, idx_to_remove, objective,
        selection_per_cluster, nonselection,
        closest_points_inter):
    """
    Evaluates the effect of swapping a point for a (set of) point(s) in the solution 
    without relying on an explicit instance of the Solution class.
    NOTE: This function is designed to be used with shared memory for parallel processing!
    NOTE: In the current implementation, there is no check for feasibility, so it is assumed
            that the point can be added without violating any constraints!
    """
    try:
        num_to_add = len(idxs_to_add)
    except TypeError:
        num_to_add = 1
        idxs_to_add = [idxs_to_add]
    candidate_objective = objective + (num_to_add - 1) * _selection_cost # cost for adding and removing
    cluster = _clusters[idx_to_remove]
    # Generate pool of alternative points to compare to
    new_selection = set(selection_per_cluster[cluster])
    for idx in idxs_to_add:
        new_selection.add(idx)
    new_selection.remove(idx_to_remove)
    new_nonselection = set(nonselection)
    new_nonselection.add(idx_to_remove)

    # Calculate intra cluster distances for cluster of new point
    add_within_cluster = []
    for idx in new_nonselection:
        cur_closest_distance = _closest_distances_intra[idx]
        cur_closest_point = _closest_points_intra[idx]
        if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
            cur_closest_distance = np.inf
            for other_idx in new_selection:
                cur_dist = get_distance(idx, other_idx, _distances, _num_points)
                if cur_dist < cur_closest_distance:
                    cur_closest_distance = cur_dist
                    cur_closest_point = other_idx
            candidate_objective += cur_closest_distance - _closest_distances_intra[idx]
            add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
        else: #point to be removed is not closest, check if newly added point is closer
            cur_dists = [(get_distance(idx, idx_to_add, _distances, _num_points), idx_to_add) for idx_to_add in idxs_to_add]
            cur_dist, idx_to_add = min(cur_dists, key = lambda x: x[0])
            if cur_dist < cur_closest_distance:
                candidate_objective += cur_dist - cur_closest_distance
                add_within_cluster.append((idx, idx_to_add, cur_dist))
    # Calculate intra cluster distances for all other clusters
    add_for_other_clusters = []
    for other_cluster in _unique_clusters:
        if other_cluster != cluster:
            cur_closest_similarity = _closest_distances_inter[cluster, other_cluster]
            if other_cluster < cluster:
                cur_closest_point = closest_points_inter[other_cluster, cluster][1]
            else:
                cur_closest_point = closest_points_inter[cluster, other_cluster][0]
            cur_closest_pair = (-1, -1)
            if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                cur_closest_similarity = -np.inf
                for idx in selection_per_cluster[other_cluster]:
                    for other_idx in new_selection:
                        cur_similarity = 1.0 - get_distance(idx, other_idx, _distances, _num_points)
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            if other_cluster < cluster:
                                cur_closest_pair = (idx, other_idx)
                            else:
                                cur_closest_pair = (other_idx, idx)
                candidate_objective += cur_closest_similarity - _closest_distances_inter[cluster, other_cluster]
                add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
            else: #point to be removed is not closest, check if one of newly added points is closer
                for idx in selection_per_cluster[other_cluster]:
                    cur_similarities = [(1.0 - get_distance(idx, idx_to_add, _distances, _num_points), idx_to_add) for idx_to_add in idxs_to_add]
                    cur_similarity, idx_to_add = max(cur_similarities, key = lambda x: x[0])
                    if cur_similarity > cur_closest_similarity:
                        cur_closest_similarity = cur_similarity
                        if other_cluster < cluster:
                            cur_closest_pair = (idx, idx_to_add)
                        else:
                            cur_closest_pair = (idx_to_add, idx)
                if cur_closest_pair[0] > -1:
                    candidate_objective += cur_closest_similarity - _closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
    if candidate_objective < objective and np.abs(candidate_objective - objective) > 1e-5:
        return candidate_objective, add_within_cluster, add_for_other_clusters
    else:
        return -1, -1, -1  # -1, -1, -1 to signify no improvement

def evaluate_remove_mp(
        idx_to_remove, objective,
        selection_per_cluster, nonselection,
        closest_points_inter):
    """
    Evaluates the effect of removing a point from the solution without relying on an explicit instance
    of the Solution class.
    NOTE: This function is designed to be used with shared memory for parallel processing!
    NOTE: In the current implementation, there is no check for feasibility, so it is assumed
            that the point can be added without violating any constraints!
    """
    candidate_objective = objective - _selection_cost # cost for removing the point
    cluster = _clusters[idx_to_remove]
    # Generate pool of alternative points to compare to
    new_selection = set(selection_per_cluster[cluster])
    new_selection.remove(idx_to_remove)
    new_nonselection = set(nonselection)
    new_nonselection.add(idx_to_remove)
    # Unlike other evaluate functions, we start by checking the inter-cluster distances first since
    # removing a point can only decrease the inter-cluster cost.

    # Calculate inter cluster distances for all other clusters
    add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
    for other_cluster in _unique_clusters:
        if other_cluster != cluster:
            cur_closest_similarity = _closest_distances_inter[cluster, other_cluster]
            if other_cluster < cluster:
                cur_closest_point = closest_points_inter[other_cluster, cluster][1]
            else:
                cur_closest_point = closest_points_inter[cluster, other_cluster][0]
            cur_closest_pair = (-1, -1)
            if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                cur_closest_similarity = -np.inf
                for idx in selection_per_cluster[other_cluster]:
                    for other_idx in new_selection:
                        cur_similarity = 1.0 - get_distance(idx, other_idx, _distances, _num_points)
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            if other_cluster < cluster:
                                cur_closest_pair = (idx, other_idx)
                            else:
                                cur_closest_pair = (other_idx, idx)
                candidate_objective += cur_closest_similarity - _closest_distances_inter[cluster, other_cluster]
                add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
    
    if candidate_objective > objective:
        return -1, -1, -1
    
    # Calculate intra cluster distances for cluster of removed point
    add_within_cluster = [] #this stores changes that have to be made if the objective improves
    for idx in new_nonselection:
        cur_closest_distance = _closest_distances_intra[idx]
        cur_closest_point = _closest_points_intra[idx]
        if cur_closest_point == idx_to_remove:
            cur_closest_distance = np.inf
            for other_idx in new_selection:
                cur_dist = get_distance(idx, other_idx, _distances, _num_points)
                if cur_dist < cur_closest_distance:
                    cur_closest_distance = cur_dist
                    cur_closest_point = other_idx
            candidate_objective += cur_closest_distance - _closest_distances_intra[idx]
            add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))

    if candidate_objective < objective and np.abs(candidate_objective - objective) > 1e-5:
        return candidate_objective, add_within_cluster, add_for_other_clusters
    else:
        return -1, -1, -1

def evaluate_swap_helper_shm(
        idxs_to_add, idx_to_remove,
        distances, cluster, unique_clusters, objective, selection_cost, num_points,
        selection_per_cluster, nonselection_per_cluster,
        closest_distances_intra, closest_distances_inter,
        closest_points_intra, closest_points_inter):
    """
    Evaluates the effect of swapping a point for a (set of) point(s) in the solution 
    without relying on an explicit instance of the Solution class.
    NOTE: This function is designed to be used with shared memory for parallel processing!
    NOTE: In the current implementation, there is no check for feasibility, so it is assumed
            that the point can be added without violating any constraints!
    Parameters:
    -----------
    idxs_to_add: int or list of int
        The index or indices of the point(s) to be added.
    idx_to_remove: int
        The index of the point to be removed.
    distances: tuple
        A tuple containing:
            - shared memory name for the distances matrix.
            - shape of the distance matrix.
            - dtype of the distance matrix.
    clusters: np.ndarray
        The cluster assignments for each point.
    unique_clusters: np.ndarray
        The unique clusters present in the solution.
    objective: float
        The current objective value of the solution.
    selection_cost: float
        The cost associated with selecting a point.
    num_points: int
        The total number of points in the dataset.
    selection_per_cluster: dict
        A dictionary mapping each cluster to a set of indices of selected points in that cluster.
    nonselection_per_cluster: dict
        A dictionary mapping each cluster to a set of indices of non-selected points in that cluster.
    closest_distances_intra: np.ndarray
        A 1D array containing the closest intra-cluster distances for each point.
    closest_distances_inter: np.ndarray
        An array containing the closest inter-cluster distances between clusters.
    closest_points_intra: np.ndarray
        A 1D array containing the closest points within each cluster for each point.
    closest_points_inter: dict
        A dictionary mapping each pair of clusters to their closest points (as tuples).
        NOTE: The key tuples are assumed to always be ordered with the smaller cluster index first.
    Returns:
    --------
    candidate_objective: float
        The objective value if the swap is made.
    add_within_cluster: list
        A list of tuples (index, closest_point, distance) for points within the cluster of the new point that would be updated.
    add_for_other_clusters: list
        A list of tuples (other_cluster, closest_pair, similarity) for points in other clusters that would be updated.
    """
    # Check if a single index, or a list of indices is provided
    try:
        num_to_add = len(idxs_to_add)
    except TypeError:
        num_to_add = 1
        idxs_to_add = [idxs_to_add]
    candidate_objective = objective + (num_to_add - 1) * selection_cost # cost for adding and removing
    #cluster = clusters[idx_to_remove]
    # Generate pool of alternative points to compare to
    new_selection = set(selection_per_cluster[cluster])
    for idx in idxs_to_add:
        new_selection.add(idx)
    new_selection.remove(idx_to_remove)
    new_nonselection = set(nonselection_per_cluster[cluster])
    new_nonselection.add(idx_to_remove)
    try:
        D = shm.SharedMemory(name=distances[0])
        D_array = np.ndarray(distances[1], dtype=np.float32, buffer=D.buf)

        # Calculate intra cluster distances for cluster of new point
        #   - check if removed point was closest selected point for any of the unselected points -> if so, replace with new point
        #   - check if added point(s) is closest selected point for any of the unselected points -> if so, replace
        add_within_cluster = []
        for idx in new_nonselection:
            cur_closest_distance = closest_distances_intra[idx]
            cur_closest_point = closest_points_intra[idx]
            if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                cur_closest_distance = np.inf
                for other_idx in new_selection:
                    cur_dist = get_distance(idx, other_idx, D_array, num_points)
                    if cur_dist < cur_closest_distance:
                        cur_closest_distance = cur_dist
                        cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
            else: #point to be removed is not closest, check if newly added point is closer
                cur_dists = [(get_distance(idx, idx_to_add, D_array, num_points), idx_to_add) for idx_to_add in idxs_to_add]
                cur_dist, idx_to_add = min(cur_dists, key = lambda x: x[0])
                if cur_dist < cur_closest_distance:
                    candidate_objective += cur_dist - cur_closest_distance
                    add_within_cluster.append((idx, idx_to_add, cur_dist))

        # Calculate intra cluster distances for all other clusters
        add_for_other_clusters = []
        for other_cluster in unique_clusters:
            if other_cluster != cluster:
                cur_closest_similarity = get_distance(cluster, other_cluster, closest_distances_inter, unique_clusters.shape[0])
                if other_cluster < cluster:
                    cur_closest_point = closest_points_inter[other_cluster, cluster][1]
                else:
                    cur_closest_point = closest_points_inter[cluster, other_cluster][0]
                cur_closest_pair = (-1, -1) #from -> to (from perspective of "other_cluster")
                if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                    cur_closest_similarity = -np.inf
                    for idx in selection_per_cluster[other_cluster]:
                        for other_idx in new_selection:
                            cur_similarity = 1.0 - get_distance(idx, other_idx, D_array, num_points)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                if other_cluster < cluster:
                                    cur_closest_pair = (idx, other_idx)
                                else:
                                    cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - get_distance(cluster, other_cluster, closest_distances_inter, unique_clusters.shape[0])
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
                else: #point to be removed is not closest, check if one of newly added points is closer
                    for idx in selection_per_cluster[other_cluster]:
                        cur_similarities = [(1.0 - get_distance(idx, idx_to_add, D_array, num_points), idx_to_add) for idx_to_add in idxs_to_add]
                        cur_similarity, idx_to_add = max(cur_similarities, key = lambda x: x[0])
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            if other_cluster < cluster:
                                cur_closest_pair = (idx, idx_to_add)
                            else:
                                cur_closest_pair = (idx_to_add, idx)
                    if cur_closest_pair[0] > -1:
                        candidate_objective += cur_closest_similarity - get_distance(cluster, other_cluster, closest_distances_inter, unique_clusters.shape[0])
                        add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))    

        return candidate_objective, add_within_cluster, add_for_other_clusters
    except Exception as e:
        print(f"Error in evaluate_swap_helper_shm: {e}")
        return None, None, None
    finally:
        D.close()

def evaluate_remove_helper_shm(
        idx_to_remove,
        distances, cluster, unique_clusters, objective, selection_cost, num_points,
        selection_per_cluster, nonselection_per_cluster,
        closest_distances_intra, closest_distances_inter,
        closest_points_intra, closest_points_inter):
    """
    Evaluates the effect of removing a point from the solution without relying on an explicit instance
    of the Solution class.
    NOTE: This function is designed to be used with shared memory for parallel processing!
    NOTE: In the current implementation, there is no check for feasibility, so it is assumed
            that the point can be added without violating any constraints!
    Parameters:
    -----------
    idx_to_remove: int
        The index of the point to be added.
    distances: tuple
        A tuple containing:
            - shared memory name for the distances matrix.
            - shape of the distance matrix.
            - dtype of the distance matrix.
    clusters: np.ndarray
        The cluster assignments for each point.
    unique_clusters: np.ndarray
        The unique clusters present in the solution.
    objective: float
        The current objective value of the solution.
    selection_cost: float
        The cost associated with selecting a point.
    num_points: int
        The total number of points in the dataset.
    selection_per_cluster: dict
        A dictionary mapping each cluster to a set of indices of selected points in that cluster.
    nonselection_per_cluster: dict
        A dictionary mapping each cluster to a set of indices of non-selected points in that cluster.
    closest_distances_intra: np.ndarray
        A 1D array containing the closest intra-cluster distances for each point.
    closest_distances_inter: np.ndarray
        An array containing the closest inter-cluster distances between clusters.
    closest_points_intra: np.ndarray
        A 1D array containing the closest points within each cluster for each point.
    closest_points_inter: dict
        A dictionary mapping each pair of clusters to their closest points (as tuples).
        NOTE: The key tuples are assumed to always be ordered with the smaller cluster index first.
    Returns:
    --------
    candidate_objective: float
        The objective value if the point is removed.
    add_within_cluster: list
        A list of tuples (index, closest_point, distance) for points within the cluster of the new point that would be updated.
    add_for_other_clusters: list
        A list of tuples (other_cluster, closest_pair, similarity) for points in other clusters that would be updated.
    """
    #cluster = clusters[idx_to_remove]
    candidate_objective = objective - selection_cost # cost for removing the point
    new_selection = set(selection_per_cluster[cluster])
    new_selection.remove(idx_to_remove)
    new_nonselection = set(nonselection_per_cluster[cluster])
    new_nonselection.add(idx_to_remove)
    try:
        D = shm.SharedMemory(name=distances[0])
        D_array = np.ndarray(distances[1], dtype=np.float32, buffer=D.buf)
        # Calculate intra cluster distances for cluster of removed point
        #   - Check if removed point was closest selected point for any of the unselected points -> if so, replace with new point
        add_within_cluster = []
        for idx in new_nonselection:
            cur_closest_distance = closest_distances_intra[idx]
            cur_closest_point = closest_points_intra[idx]
            if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                cur_closest_distance = np.inf
                for other_idx in new_selection:
                    cur_dist = get_distance(idx, other_idx, D_array, num_points)
                    if cur_dist < cur_closest_distance:
                        cur_closest_distance = cur_dist
                        cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))

        # Calculate inter cluster distances for all other clusters
        #  - Check if removed point was closest selected point for any of the other clusters -> if so replace with another point (looping over all selected points in cluster)
        add_for_other_clusters = []
        for other_cluster in unique_clusters:
            if other_cluster != cluster:
                cur_closest_similarity = get_distance(cluster, other_cluster, closest_distances_inter, unique_clusters.shape[0])
                if other_cluster < cluster:
                    cur_closest_point = closest_points_inter[other_cluster, cluster][1]
                else:
                    cur_closest_point = closest_points_inter[cluster, other_cluster][0]
                cur_closest_pair = (-1, -1) #from - to (considered from perspective of "other_cluster")
                if cur_closest_point == idx_to_remove:
                    cur_closest_similarity = -np.inf
                    for idx in selection_per_cluster[other_cluster]:
                        for other_idx in new_selection:
                            cur_similarity = 1 - get_distance(idx, other_idx, D_array, num_points)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                if other_cluster < cluster:
                                    cur_closest_pair = (idx, other_idx)
                                else:
                                    cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - get_distance(cluster, other_cluster, closest_distances_inter, unique_clusters.shape[0])
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
                    
        return candidate_objective, add_within_cluster, add_for_other_clusters
    except Exception as e:
        print(f"Error in evaluate_remove_helper_shm: {e}")
        return None, None, None
    finally:
        D.close()

def init_worker(
        distances_name, distances_shape, 
        clusters_name, clusters_shape,
        closest_distances_intra_name, closest_distances_intra_shape, 
        closest_points_intra_name, closest_points_intra_shape,
        closest_distances_inter_name, closest_distances_inter_shape,
        unique_clusters, selection_cost, num_points):
    
    import numpy as np
    import multiprocessing.shared_memory as shm
    import atexit

    global _distances_shm, _distances
    _distances_shm = shm.SharedMemory(name=distances_name)
    _distances = np.ndarray(distances_shape, dtype=np.float32, buffer=_distances_shm.buf)
    global _clusters_shm, _clusters
    _clusters_shm = shm.SharedMemory(name=clusters_name)
    _clusters = np.ndarray(clusters_shape, dtype=np.int64, buffer=_clusters_shm.buf)
    global _closest_distances_intra_shm, _closest_distances_intra
    _closest_distances_intra_shm = shm.SharedMemory(name=closest_distances_intra_name)
    _closest_distances_intra = np.ndarray(closest_distances_intra_shape, dtype=np.float32, buffer=_closest_distances_intra_shm.buf)
    global _closest_points_intra_shm, _closest_points_intra
    _closest_points_intra_shm = shm.SharedMemory(name=closest_points_intra_name)
    _closest_points_intra = np.ndarray(closest_points_intra_shape, dtype=np.int32, buffer=_closest_points_intra_shm.buf)
    global _closest_distances_inter_shm, _closest_distances_inter
    _closest_distances_inter_shm = shm.SharedMemory(name=closest_distances_inter_name)
    _closest_distances_inter = np.ndarray(closest_distances_inter_shape, dtype=np.float32, buffer=_closest_distances_inter_shm.buf)
    global _unique_clusters, _selection_cost, _num_points
    _unique_clusters = unique_clusters
    _selection_cost = selection_cost
    _num_points = num_points

    def cleanup():
        try:
            _distances_shm.close()
            _clusters_shm.close()
            _closest_distances_intra_shm.close()
            _closest_points_intra_shm.close()
            _closest_distances_inter_shm.close()
        except Exception as e:
            print(f"Error closing shared memory: {e}")

    atexit.register(cleanup)

def process_batch(batch, event, closest_points_inter, selection_per_cluster, nonselection_per_cluster, objective):
    global _distances, _clusters, _closest_distances_intra, _closest_points_intra, _closest_distances_inter
    global _unique_clusters, _selection_cost, _num_points

    num_improvements = 0
    num_moves = 0
    for task, content in batch:
        if event.is_set():
            return None, None, -1, None, None
        if task == "add":
            idx_to_add = content
            cluster = _clusters[idx_to_add]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_add_mp(idx_to_add, objective, selection_per_cluster, nonselection_per_cluster[cluster])
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 0
                event.set()
                return "add", content, candidate_objective, add_within_cluster, add_for_other_clusters
        elif task == "swap" or task == "doubleswap":
            idxs_to_add, idx_to_remove = content
            cluster = _clusters[idx_to_remove]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_swap_mp(idxs_to_add, idx_to_remove, objective, selection_per_cluster, nonselection_per_cluster[cluster], closest_points_inter)
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 1
                event.set()
                return task, content, candidate_objective, add_within_cluster, add_for_other_clusters
        elif task == "remove":
            idx_to_remove = content
            cluster = _clusters[idx_to_remove]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_remove_mp(idx_to_remove, objective, selection_per_cluster, nonselection_per_cluster[cluster], closest_points_inter)
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 1
                event.set()
                return "remove", content, candidate_objective, add_within_cluster, add_for_other_clusters

    return None, None, -1, None, None

def process_batch_result(result, results_list):
    move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters = result
    if candidate_objective > -1:
        results_list.append((move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters))

def test(args, event):
    for a in args:
        if event.is_set():
            break
        else:
            x = np.random.rand()
    #print("This is being executed:", args, "\n", flush=True)

def get_index(idx1, idx2, num_points):
    """
    Returns the index in the condensed distance matrix for the given pair of indices.
    """
    if idx1 == idx2:
        return -1
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    return num_points * idx1 - (idx1 * (idx1 + 1)) // 2 + idx2 - idx1 - 1

def get_distance(idx1, idx2, distances, num_points):
    """
    Returns the distance between two points which has to be
    converted since the distance matrix is stored as a
    condensed matrix.
    """
    if idx1 == idx2:
        return 0.0
    index = get_index(idx1, idx2, num_points)
    return distances[index]

if __name__ == "__main__":
    pass