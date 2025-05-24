import numpy as np
from scipy.spatial.distance import squareform
import itertools
import math
from decimal import Decimal, getcontext


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

        self.feasible = self.determine_feasibility()

        if self.feasible:
            # Set objective value
            self.objective = np.sum(self.selection) * self.selection_cost
            # INTRA cluster distances
            for idx in np.where(~self.selection)[0]:
                cur_min = np.float32(np.inf)
                cur_idx = idx
                for other_idx in np.where((self.clusters == self.clusters[idx]) & self.selection)[0]:
                    cur_dist = self.get_distance(idx, other_idx)
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
                    cur_dist = 1.0 - self.get_distance(point_pair[0], point_pair[1]) #WARNING: precision errors might occur here!!
                    if cur_dist > cur_max:
                        cur_max = cur_dist
                        cur_pair = point_pair
                self.closest_distances_inter[cluster_pair[0], cluster_pair[1]] = cur_max
                self.closest_distances_inter[cluster_pair[1], cluster_pair[0]] = cur_max
                self.closest_points_inter[(cluster_pair[0], cluster_pair[1])] = cur_pair  # Store the first point index
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

    def evaluate_add(self, idx_to_add):
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
            cur_dist = self.get_distance(idx, idx_to_add) # distance to current point (idx)
            if cur_dist < self.closest_distances_intra[idx]:
                candidate_objective += cur_dist - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_dist))
        # Calculate inter cluster distances for all other clusters
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                cur_max = self.closest_distances_inter[cluster, other_cluster]
                cur_idx = -1
                for idx in self.selection_per_cluster[other_cluster]:
                    cur_similarity = 1 - self.get_distance(idx, idx_to_add) #this is the similarity, if it is more similar then change solution
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
                    cur_dist = self.get_distance(idx, other_idx)
                    if cur_dist < cur_closest_distance:
                        cur_closest_distance = cur_dist
                        cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
            else: #point to be removed is not closest, check if newly added point is closer
                cur_dist = self.get_distance(idx, idx_to_add)
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
                            cur_similarity = 1 - self.get_distance(idx, other_idx)
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
                        cur_similarity = 1 - self.get_distance(idx, idx_to_add)
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
                    cur_dist = self.get_distance(idx, other_idx)
                    if cur_dist < cur_closest_distance:
                        cur_closest_distance = cur_dist
                        cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
            else: #point to be removed is not closest, check if one of newly added points is closer
                cur_dist1 = self.get_distance(idx, idx_to_add1)
                cur_dist2 = self.get_distance(idx, idx_to_add2)
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
                            cur_similarity = 1 - self.get_distance(idx, other_idx)
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
                        cur_similarity1 = 1 - self.get_distance(idx, idx_to_add1)
                        cur_similarity2 = 1 - self.get_distance(idx, idx_to_add2)
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

    def evaluate_remove(self, idx_to_remove):
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
                        cur_dist = self.get_distance(idx, other_idx)
                        if cur_dist < cur_closest_distance:
                            cur_closest_distance = cur_dist
                            cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
        # Calculate inter cluster distances for all other clusters
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
                            cur_similarity = 1 - self.get_distance(idx, other_idx)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                if other_cluster < cluster:
                                    cur_closest_pair = (idx, other_idx)
                                else:
                                    cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
        
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

        if type(seed) is int:
            np.random.seed(seed)

        solution_changed = False
        while iteration < max_iterations:
            solution_changed = False
            for move_type, move_content in self.generate_random_moves():
                if move_type == "add":
                    idx_to_add = move_content
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_add(idx_to_add, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        objectives.append((self.objective, f"added {idx_to_add} from cluster {self.clusters[idx_to_add]}"))
                        selections.append(self.selection.copy())
                        break
                elif move_type == "swap":
                    idx_to_add, idx_to_remove = move_content
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_swap(idx_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        objectives.append((self.objective, f"swapped {idx_to_remove} for {idx_to_add} in cluster {self.clusters[idx_to_add]}"))
                        selections.append(self.selection.copy())
                        break
                elif move_type == "doubleswap":
                    (idx_to_add1, idx_to_add2), idx_to_remove = move_content
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > 1e-8:
                        self.accept_doubleswap((idx_to_add1, idx_to_add2), idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                        solution_changed = True
                        objectives.append((self.objective, f"doubleswapped {idx_to_remove} for {idx_to_add1} and {idx_to_add2} in cluster {self.clusters[idx_to_add1]}"))
                        selections.append(self.selection.copy())
                        break
                elif move_type == "remove":
                    idx_to_remove = move_content
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

    def generate_random_moves(self):
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

        while active_generators:
            # Randomly pick an active generator
            selected_generator = np.random.choice(active_generators)
            try:
                # Yield the next value from the selected generator
                yield selected_generator, next(generators[selected_generator])
            except StopIteration:
                # Remove the generator if it is exhausted
                active_generators.remove(selected_generator)

    def get_distance(self, idx1, idx2):
        """
        Returns the distance between two points which has to be
        converted since the distance matrix is stored as a
        condensed matrix.
        """
        if idx1 == idx2:
            return 0.0
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        return self.distances[self.num_points * idx1 - (idx1 * (idx1 + 1)) // 2 + idx2 - idx1 - 1]