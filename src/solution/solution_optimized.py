import numpy as np
from scipy.spatial.distance import squareform
import itertools
import math
import multiprocessing.shared_memory as shm
from multiprocessing import Pool, Manager
import time
import traceback

# This is to define the precision threshold for floating point comparisons
PRECISION_THRESHOLD = 1e-6

class Solution:
    def __init__(self, distances: np.ndarray, clusters: np.ndarray, selection=None, selection_cost: float = 1.0, cost_per_cluster: int = 0, seed=None):
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

        # Set random state for reproducibility
        if isinstance(seed, int):
            self.random_state = np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            self.random_state = seed
        else:
            self.random_state = np.random.RandomState()

        # Initialize object attributes
        self.selection = selection.astype(dtype=bool)
        self.distances = squareform(distances.astype(dtype=np.float32))
        self.clusters = clusters.astype(dtype=np.int64)
        self.unique_clusters = np.unique(self.clusters)
        # Cost per cluster based on number of points in each cluster
        # If cost_per_cluster is True, then the cost is divided by the number of points in each cluster
        # cost_per_cluster is indexed by cluster indices
        self.selection_cost = selection_cost
        self.cost_per_cluster = np.zeros(self.unique_clusters.shape[0], dtype=np.float64)
        for cluster in self.unique_clusters:
            if cost_per_cluster == 0: #default behavior, set to selection cost
                self.cost_per_cluster[cluster] = selection_cost
            elif cost_per_cluster == 1: #set to 1 / number of points in cluster
                self.cost_per_cluster[cluster] = selection_cost / np.sum(self.clusters == cluster)
            elif cost_per_cluster == 2:
                # Define the average distance in a cluster as the average distance
                # of all points in the cluster to the centroid of the cluster.
                cluster_points = np.where(self.clusters == cluster)[0]
                centroid = np.argmin(np.sum(distances[np.ix_(cluster_points, cluster_points)], axis=1))
                self.cost_per_cluster[cluster] = np.mean(distances[centroid, cluster_points])
            elif cost_per_cluster == 3:
                # Define the average distance in a cluster as the average distance
                # of all points in the cluster to the closest point in the cluster.
                cluster_points = np.where(self.clusters == cluster)[0]
                self.cost_per_cluster[cluster] = np.mean([np.min(distances[point, cluster_points]) for point in cluster_points])
        self.num_points = distances.shape[0]

        # Process initial representation to optimize for comparisons speed
        self.points_per_cluster = {cluster: set(np.where(self.clusters == cluster)[0]) for cluster in self.unique_clusters} #points in every cluster

        self.calculate_objective()
        
    def __eq__(self, other):
        """
        Check if two solutions are equal.
        """
        if not isinstance(other, Solution):
            print("Other object is not a Solution instance.")
            return False
        # Check if distances and clusters are equal
        if not np.allclose(self.distances, other.distances, atol=PRECISION_THRESHOLD) or not np.array_equal(self.clusters, other.clusters):
            print("Distances or clusters are not equal.")
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
        if not math.isclose(self.selection_cost, other.selection_cost, rel_tol=PRECISION_THRESHOLD):
            return False
        # Check if feasible is equal
        if self.feasible != other.feasible:
            return False
        if self.feasible:
            # Check if objective is equal
            if not math.isclose(self.objective, other.objective, rel_tol=PRECISION_THRESHOLD):
                print("Objective values are not equal: ", self.objective, other.objective)
                return False
            # Check if closest_distances_intra is equal
            if not np.allclose(self.closest_distances_intra, other.closest_distances_intra, atol=PRECISION_THRESHOLD):
                print("Closest distances intra are not equal.")
                return False
            # Check if closest_points_intra is equal
            if not np.allclose(self.closest_points_intra, other.closest_points_intra, atol=PRECISION_THRESHOLD):
                print("Closest points intra are not equal.")
                return False
            # Check if closest_distances_inter is equal
            if not np.allclose(self.closest_distances_inter, other.closest_distances_inter, atol=PRECISION_THRESHOLD):
                print("Closest distances inter are not equal.")
                return False
            # Check if closest_points_inter is equal
            if set(self.closest_points_inter.keys()) != set(other.closest_points_inter.keys()):
                print("Closest points inter keys are not equal.")
                return False
            for key in self.closest_points_inter:
                if self.closest_points_inter[key] != other.closest_points_inter[key]:
                    print(f"Closest points inter for key {key} are not equal.")
                    return False
            if not np.allclose(self.closest_points_inter_array, other.closest_points_inter_array, atol=PRECISION_THRESHOLD):
                print("Closest points inter array are not equal.")
                return False
        return True

    def determine_feasibility(self):
        """
        Determines if the solution stored in this object is feasible.
        NOTE: A solution is feasible if every cluster has at least one selected point.
        """
        uncovered_clusters = set(self.unique_clusters)
        for point in np.where(self.selection)[0]:
            uncovered_clusters.discard(self.clusters[point])
        return len(uncovered_clusters) == 0
    
    def calculate_objective(self):
        """
        Calculates the objective value of the solution, as well as set all the
        inter and intra cluster distances and points.
        NOTE: If selection is not feasible, the objective value is set to np.inf
        and some of the internal attributes will not be set.
        """
        # Re-determine the selected and unselected points for every cluster
        self.selection_per_cluster = {cluster: set(np.where((self.clusters == cluster) & self.selection)[0]) for cluster in self.unique_clusters} #selected points in every cluster
        self.nonselection_per_cluster = {cluster: set(np.where((self.clusters == cluster) & ~self.selection)[0]) for cluster in self.unique_clusters} #unselected points in every cluster
        
        # Re-initialize the closest distances and points arrays and dicts
        # INTRA CLUSTER INFORMATION
        self.closest_distances_intra = np.zeros(self.selection.shape[0], dtype=np.float32) #distances to closest selected point
        self.closest_points_intra = np.arange(0, self.selection.shape[0], dtype=np.int32) #indices of closest selected point
        # INTER CLUSTER INFORMATION
        self.closest_distances_inter = np.zeros((self.unique_clusters.shape[0], self.unique_clusters.shape[0]), dtype=np.float32) #distances to closest selected point
        self.closest_points_inter = {pair: (None, None) for pair in itertools.combinations(self.unique_clusters, 2)} #indices of closest selected point
        """
        Interpretation of closest_points_inter_array: given a pair of clusters (cluster1, cluster2),
        the value at closest_points_inter_array[cluster1, cluster2] is the index of the point in cluster2 that is closest to any point in cluster1.
        The value at closest_points_inter_array[cluster2, cluster1] is the index of the point in cluster1 that is closest to any point in cluster2.
        """
        self.closest_points_inter_array = np.zeros((self.unique_clusters.shape[0], self.unique_clusters.shape[0]), dtype=np.int32)

        is_feasible = self.determine_feasibility()
        if not is_feasible:
            self.feasible = False
            self.objective = np.inf
            print("The solution is infeasible, objective value is set to INF and the closest distances & points are not set.")
            return self.objective
        self.feasible = True

        # Calculate the objective value
        objective = 0.0
        # Selection cost
        for idx in np.where(self.selection)[0]:
            objective += self.cost_per_cluster[self.clusters[idx]]
        # Intra cluster distance costs
        for cluster in self.unique_clusters:
            for idx in self.nonselection_per_cluster[cluster]:
                cur_min = np.float32(np.inf)
                cur_idx = None # index of the closest selected point of the same cluster
                for other_idx in sorted(list(self.selection_per_cluster[cluster])): #this is to ensure consistent ordering
                    cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                    if cur_dist < cur_min:
                        cur_min = cur_dist
                        cur_idx = other_idx
                self.closest_distances_intra[idx] = np.float32(cur_min)
                self.closest_points_intra[idx] = np.int32(cur_idx)
                objective += cur_min
        # Inter cluster distance costs
        for cluster_1, cluster_2 in itertools.combinations(self.unique_clusters, 2):
            cur_max = -np.float32(np.inf)
            cur_pair = (None, None) # indices of the closest selected points of the two clusters
            for point_1 in sorted(list(self.selection_per_cluster[cluster_1])): #this is to ensure consistent ordering
                for point_2 in sorted(list(self.selection_per_cluster[cluster_2])): #this is to ensure consistent ordering
                    cur_dist = 1.0 - get_distance(point_1, point_2, self.distances, self.num_points)
                    if cur_dist > cur_max:
                        cur_max = cur_dist
                        cur_pair = (point_1, point_2)
            self.closest_distances_inter[cluster_1, cluster_2] = cur_max
            self.closest_distances_inter[cluster_2, cluster_1] = cur_max
            self.closest_points_inter[(cluster_1, cluster_2)] = cur_pair
            self.closest_points_inter_array[(cluster_1, cluster_2)] = cur_pair[1]
            self.closest_points_inter_array[(cluster_2, cluster_1)] = cur_pair[0]
            objective += cur_max
        self.objective = objective

    @staticmethod
    def generate_centroid_solution(distances, clusters, selection_cost: float = 1.0, cost_per_cluster: int = 0, seed=None):
        """
        Generates a Solution object with an initial solution by selecting the centroid for every cluster.

        Parameters:
        -----------
        distances: numpy.ndarray
            A 2D array where distances[i, j] represents the distance (similarity) between point i and point j.
            NOTE: distances should be in the range [0, 1].
        clusters: numpy.ndarray
            A 1D array where clusters[i] represents the cluster assignment of point i.
        selection_cost: float
            The cost associated with selecting a point.
        cost_per_cluster: int
            Defines how the cost of selecting a point in each cluster is calculated.
            0: Default behavior, set to selection cost.
            1: Set to selection_cost / number of points in cluster.
            2: Set to the average distance in a cluster (average distance of all points in the cluster to the centroid of the cluster).
            3: Set to the average distance in a cluster (average distance of all points in the cluster to the closest point in the cluster).
        seed: int, optional
            Random seed for reproducibility.
            NOTE: The seed will create a random state for the solution, which is used for
            operations that introduce stochasticity, such as random selection of points.

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

        return Solution(distances, clusters, selection=selection, selection_cost=selection_cost, cost_per_cluster=cost_per_cluster, seed=seed)
    
    @staticmethod
    def generate_random_solution(distances, clusters, selection_cost: float = 1.0, cost_per_cluster: int = 0, max_fraction=0.1, seed=None):
        """
        Generates a Solution object with an initial solution by randomly selecting points.

        Parameters:
        -----------
        distances: numpy.ndarray
            A 2D array where distances[i, j] represents the distance (similarity) between point i and point j.
            NOTE: distances should be in the range [0, 1].
        clusters: numpy.ndarray
            A 1D array where clusters[i] represents the cluster assignment of point i.
        selection_cost: float
            The cost associated with selecting a point.
        cost_per_cluster: int
            Defines how the cost of selecting a point in each cluster is calculated.
            0: Default behavior, set to selection cost.
            1: Set to selection_cost / number of points in cluster.
            2: Set to the average distance in a cluster (average distance of all points in the cluster to the centroid of the cluster).
            3: Set to the average distance in a cluster (average distance of all points in the cluster to the closest point in the cluster).
        max_fraction: float
            The maximum fraction of points to select (0-1].
            NOTE: If smaller than 1 divided by the number of clusters,
            at least one point per cluster will be selected.
        seed: int, optional
            Random seed for reproducibility.
            NOTE: The seed will create a random state for the solution which is used for
            operations that introduce stochasticity, such as random selection of points.

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

        if isinstance(seed, int):
            random_state = np.random.RandomState(seed)
        else:
            random_state = np.random.RandomState()

        # Ensure at least one point per cluster is selected
        for cluster in unique_clusters:
            cluster_points = np.where(clusters == cluster)[0]
            selected_point = random_state.choice(cluster_points)
            selection[selected_point] = True

        # Randomly select additional points up to the max_fraction limit
        num_points = clusters.shape[0]
        max_selected_points = int(max_fraction * num_points)
        remaining_points = np.where(~selection)[0]
        num_additional_points = max(0, max_selected_points - np.sum(selection))
        additional_points = random_state.choice(remaining_points, size=num_additional_points, replace=False)
        selection[additional_points] = True

        return Solution(distances, clusters, selection, selection_cost=selection_cost, cost_per_cluster=cost_per_cluster, seed=random_state)

    def evaluate_add(self, idx_to_add, local_search=False):
        """
        Evaluates the effect of adding an unselected point to the solution.

        Parameters:
        -----------
        idx_to_add: int
            The index of the point to be added.
        local_search: bool
            If True, the method will return (np.inf, None, None) if the candidate objective
            is worse than the current objective, allowing for local search to skip unnecessary evaluations.
            If False, it will always evaluate the addition.
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the addition.
            NOTE: If local_search is True, the returned value can be np.inf if
            the candidate objective is worse than the current objective based
            on intra distances.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, new_distance, index_in_other_cluster)]
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        if self.selection[idx_to_add]:
            raise ValueError("The point to add must not be selected.")
        cluster = self.clusters[idx_to_add]
        candidate_objective = self.objective + self.cost_per_cluster[cluster] # cost for adding the point
        # Calculate intra cluster distances for cluster of new point
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for idx in self.nonselection_per_cluster[cluster]:
            cur_dist = get_distance(idx, idx_to_add, self.distances, self.num_points) # distance to current point (idx)
            if cur_dist < self.closest_distances_intra[idx]:
                candidate_objective += cur_dist - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_dist))

        # NOTE: Inter-cluster distances can only increase when adding a point, so when doing local search we can exit here if objective is worse
        if candidate_objective > self.objective and np.abs(self.objective - candidate_objective) > PRECISION_THRESHOLD and local_search:
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

    def accept_add(self, idx_to_add: int, candidate_objective: float, add_within_cluster: list, add_for_other_clusters: list):
        """
        Accepts the addition of a point to the solution.
        NOTE: this assumes that the initial solution
        was feasible.

        PARAMETERS:
        -----------
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
            self.closest_points_inter_array[cluster, other_cluster] = idx
            self.closest_points_inter_array[other_cluster, cluster] = idx_to_add
        # Update objective value
        self.objective = candidate_objective
         
    def evaluate_swap(self, idx_to_add: int, idx_to_remove: int):
        """
        Evaluates the effect of swapping a selected point for an unselected point 
        in the solution.

        Parameters:
        -----------
        idx_to_add: int
            The index of the point to be added.
        idx_to_remove: int
            The index of the point to be removed.
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the addition.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, closest_point_pair, new_distance)]
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        if self.selection[idx_to_add]:
            raise ValueError("The point to add must not be selected.")
        if not self.selection[idx_to_remove]:
            raise ValueError("The point to remove must be selected.")
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

    def accept_swap(self, idx_to_add: int, idx_to_remove: int, candidate_objective: float, add_within_cluster: list, add_for_other_clusters: list):
        """
        Accepts the swap of the pair of points in the solution.
        NOTE: this assumes that the initial solution
        was feasible.

        PARAMETERS:
        -----------
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
            NOTE: for cur_closest_pair, the first index is in the cluster with lowest index.
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
                self.closest_points_inter_array[(other_cluster, cluster)] = cur_closest_pair[1]
                self.closest_points_inter_array[(cluster, other_cluster)] = cur_closest_pair[0]
            else:
                self.closest_points_inter[(cluster, other_cluster)] = cur_closest_pair
                self.closest_points_inter_array[(cluster, other_cluster)] = cur_closest_pair[1]
                self.closest_points_inter_array[(other_cluster, cluster)] = cur_closest_pair[0]
        # Update objective value
        self.objective = candidate_objective

    def evaluate_doubleswap(self, idxs_to_add, idx_to_remove: int):
        """
        Evaluates the effect of swapping a selected point for a pair of 
        unselected points in the solution.

        Parameters:
        -----------
        idxs_to_add: tuple of int or list of int
            A tuple containing the indices of the two points to be added.
        idx_to_remove: int
            The index of the point to be removed.
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the addition.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, closest_point_pair, new_distance)]
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        idx_to_add1, idx_to_add2 = idxs_to_add
        if self.selection[idx_to_add1]:
            raise ValueError("The first point to add must not be selected.")
        if self.selection[idx_to_add2]:
            raise ValueError("The second point to add must not be selected.")
        if not self.selection[idx_to_remove]:
            raise ValueError("The point to remove must be selected.")
        cluster = self.clusters[idx_to_add1]
        if cluster != self.clusters[idx_to_remove] or cluster != self.clusters[idx_to_add2]:
            raise ValueError("All points must be in the same cluster.")
        
        candidate_objective = self.objective + self.cost_per_cluster[cluster]
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

    def accept_doubleswap(self, idxs_to_add: tuple, idx_to_remove: int, candidate_objective: float, add_within_cluster: list, add_for_other_clusters: list):
        """
        Accepts the doubleswap of the pair of points in the solution.
        NOTE: this assumes that the initial solution
        was feasible.

        PARAMETERS:
        -----------
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
            NOTE: for cur_closest_pair, the first index is in the cluster with lowest index.
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
                self.closest_points_inter_array[(other_cluster, cluster)] = cur_closest_pair[1]
                self.closest_points_inter_array[(cluster, other_cluster)] = cur_closest_pair[0]
            else:
                self.closest_points_inter[(cluster, other_cluster)] = cur_closest_pair
                self.closest_points_inter_array[(cluster, other_cluster)] = cur_closest_pair[1]
                self.closest_points_inter_array[(other_cluster, cluster)] = cur_closest_pair[0]
        # Update objective value
        self.objective = candidate_objective

    def evaluate_remove(self, idx_to_remove: int, local_search: bool = False):
        """
        Evaluates the effect of removing a selected point from the solution.

        Parameters:
        -----------
        idx_to_remove: int
            The index of the point to be removed.
        local_search: bool
            If True, the method will return (np.inf, None, None) if the candidate objective
            is worse than the current objective, allowing for local search to skip unnecessary evaluations.
            If False, it will always evaluate the addition.
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the addition.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, closest_point_pair, new_distance)]
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate removal.")
        if not self.selection[idx_to_remove]:
            raise ValueError("The point to remove must be selected.")
        cluster = self.clusters[idx_to_remove]
        candidate_objective = self.objective - self.cost_per_cluster[cluster]

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

    def accept_remove(self, idx_to_remove: int, candidate_objective: float, add_within_cluster: list, add_for_other_clusters: list):
        """
        Accepts the addition of a point to the solution.
        NOTE: this assumes that the initial solution
        was feasible.

        PARAMETERS:
        -----------
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
            NOTE: for cur_closest_pair, the first index is in the cluster with lowest index.
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
                self.closest_points_inter_array[(other_cluster, cluster)] = cur_closest_pair[1]
                self.closest_points_inter_array[(cluster, other_cluster)] = cur_closest_pair[0]
            else:
                self.closest_points_inter[(cluster, other_cluster)] = cur_closest_pair
                self.closest_points_inter_array[(cluster, other_cluster)] = cur_closest_pair[1]
                self.closest_points_inter_array[(other_cluster, cluster)] = cur_closest_pair[0]
        # Update objective value
        self.objective = candidate_objective

    def accept_move(self, move_type: str, move_content, candidate_objective: float, add_within_cluster: list, add_for_other_clusters: list):
        """
        Accepts a move to the solution.
        NOTE: This assumes that the initial solution and the move
        are feasible and will not check for this.

        PARAMETERS:
        -----------
        move_type: str
            The type of the move (e.g., "add", "swap", "doubleswap", "remove").
        move_content: int or tuple
            The content of the move (e.g., index to add, indices to swap).
        candidate_objective: float
            The objective value of the solution after the move.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, cur_closest_pair, new_distance)]
            Note that for cur_closest_pair, the first index is in the cluster with lowest index.
        """
        if move_type == "add":
            idx_to_add = move_content
            self.accept_add(idx_to_add, candidate_objective, add_within_cluster, add_for_other_clusters)
        elif move_type == "swap":
            idx_to_add, idx_to_remove = move_content
            self.accept_swap(idx_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
        elif move_type == "doubleswap":
            idxs_to_add, idx_to_remove = move_content
            self.accept_doubleswap(idxs_to_add, idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
        elif move_type == "remove":
            idx_to_remove = move_content
            self.accept_remove(idx_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)

    def local_search(self, max_iterations: int = 10_000, num_cores: int = 1, hybrid: bool = True,
                           random_move_order: bool = True, random_index_order: bool = True, move_order: list = ["add", "swap", "doubleswap", "remove"],
                           batch_size: int = 1000, max_batches: int = 32, 
                           runtime_switch: float = 1.0, num_switch: int = 5,
                           logging: bool = False, logging_frequency: int = 500,
                           ):
        """
        Perform local search to find a (local) optimal solution. Allows for both
        single-core and multi-core execution.
        
        Parameters:
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        num_cores: int
            The number of cores to use for parallel processing.
            NOTE: if set to 1, local search will always run in
            single processor mode, even if hybrid is True.
        hybrid: bool
            If True, local search will switch to multiprocessing
            after num_switch consecutive iterations take longer
            than runtime_switch seconds.
        random_move_order: bool
            If True, the order of moves (add, swap, doubleswap,
            remove) is randomized.
        random_index_order: bool
            If True, the order of indices for moves is randomized.
            NOTE: if random_move_order is True, but this is false,
            all moves of a particular type will be tried before
            moving to the next move type, but the order of moves
            is random).
        move_order: list
            If provided, this list will be used to determine the
            order of moves. If random_move_order is True, this
            list will be shuffled before use.
            NOTE: this list should contain the following move types (as strings):
                - "add"
                - "swap"
                - "doubleswap"
                - "remove"
            NOTE: by leaving out a move type, it will not be
            considered in the local search.
        batch_size: int
            In multiprocessing mode, moves are processed in batches
            of this size.
            NOTE: do not set this to a value smaller than 0
        max_batches: int
            To prevent memory issues, the number of batches is
            limited to this value. Once every batch has been
            processed, the next set of batches will be
            processed.
            NOTE: this should be set to at least the number of
            num_cores, otherwise some cores will be idle.
        runtime_switch: float
            If hybrid is True, the local search will switch to
            multiprocessing after num_switch consecutive iterations
            take longer than this number of seconds.
        num_switch: int
            If hybrid is True, the local search will switch to
            multiprocessing after this number of consecutive iterations
            take longer than runtime_switch seconds.
        logging: bool
            If True, information about the local search will be printed.
        logging_frequency: int
            If logging is True, information will be printed every
            logging_frequency iterations.

        Returns:
        --------
        time_per_iteration: list of floats
            The time taken for each iteration.
            NOTE: this is primarily for logging purposes
        objectives: list of floats
            The objective value after each iteration.
        switch_iteration: int
            The iteration at which the local search switched to
            multiprocessing when using hybrid mode.
            NOTE: if the search did not switch to multiprocessing,
            or ran in single-processing mode, this will be -1.
        """
        # Validate input parameters
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(num_cores, int) or num_cores < 1:
            raise ValueError("num_cores must be a positive integer.")
        if not isinstance(hybrid, bool):
            raise ValueError("hybrid must be a boolean value.")
        if not isinstance(random_move_order, bool):
            raise ValueError("random_move_order must be a boolean value.")
        if not isinstance(random_index_order, bool):
            raise ValueError("random_index_order must be a boolean value.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(max_batches, int) or max_batches < 1:
            raise ValueError("max_batches must be a positive integer.")
        if not isinstance(runtime_switch, (int, float)) or runtime_switch < 0:
            raise ValueError("runtime_switch must be a non-negative number.")
        if not isinstance(num_switch, int) or num_switch < 1:
            raise ValueError("num_switch must be a positive integer.")
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")
        if num_cores  == 1 and hybrid:
            print("Warning: hybrid mode is set to True, but num_cores is 1. This will not use multiprocessing!")

        # Initialize variables
        iteration = 0
        time_per_iteration = []
        objectives = []
        switch_iteration = -1
        solution_changed = False
        times_runtime_exceeded = 0

        if num_cores > 1 and not hybrid:
            run_in_multiprocessing = True
        else:
            run_in_multiprocessing = False

        # Multiprocessing
        if num_cores > 1:
            try:
                # If multiprocessing is enabled, create shared memory arrays
                # Copy distance matrix to shared memory
                distances_shm = shm.SharedMemory(create=True, size=self.distances.nbytes)
                shared_distances = np.ndarray(self.distances.shape, dtype=self.distances.dtype, buffer=distances_shm.buf)
                np.copyto(shared_distances, self.distances) #this array is static, only copy once
                # Copy cluster assignment to shared memory
                clusters_shm = shm.SharedMemory(create=True, size=self.clusters.nbytes)
                shared_clusters = np.ndarray(self.clusters.shape, dtype=self.clusters.dtype, buffer=clusters_shm.buf)
                np.copyto(shared_clusters, self.clusters) #this array is static, only copy once

                # for the intra and inter distances, only copy them during iterations since they are updated during the local search
                # Copy closest_distances_intra to shared memory
                closest_distances_intra_shm = shm.SharedMemory(create=True, size=self.closest_distances_intra.nbytes)
                shared_closest_distances_intra = np.ndarray(self.closest_distances_intra.shape, dtype=self.closest_distances_intra.dtype, buffer=closest_distances_intra_shm.buf)
                # Copy closest_points_intra to shared memory
                closest_points_intra_shm = shm.SharedMemory(create=True, size=self.closest_points_intra.nbytes)
                shared_closest_points_intra = np.ndarray(self.closest_points_intra.shape, dtype=self.closest_points_intra.dtype, buffer=closest_points_intra_shm.buf)
                
                # Copy closest_distances_inter to shared memory
                closest_distances_inter_shm = shm.SharedMemory(create=True, size=self.closest_distances_inter.nbytes)
                shared_closest_distances_inter = np.ndarray(self.closest_distances_inter.shape, dtype=self.closest_distances_inter.dtype, buffer=closest_distances_inter_shm.buf)
                # Copy closest_points_inter to shared memory
                closest_points_inter_shm = shm.SharedMemory(create=True, size=self.closest_points_inter_array.nbytes)
                shared_closest_points_inter = np.ndarray(self.closest_points_inter_array.shape, dtype=self.closest_points_inter_array.dtype, buffer=closest_points_inter_shm.buf)

                with Manager() as manager:
                    event = manager.Event() #this is used to signal when tasks should be stopped
                    results = manager.list() #this is used to store an improvement is one is found

                    with Pool(
                        processes=num_cores,
                        initializer=init_worker,
                        initargs=(
                            distances_shm.name, shared_distances.shape,
                            clusters_shm.name, shared_clusters.shape,
                            closest_distances_intra_shm.name, shared_closest_distances_intra.shape,
                            closest_points_intra_shm.name, shared_closest_points_intra.shape,
                            closest_distances_inter_shm.name, shared_closest_distances_inter.shape,
                            closest_points_inter_shm.name, shared_closest_points_inter.shape,
                            self.unique_clusters, self.cost_per_cluster, self.num_points,
                        ),
                    ) as pool:
                        
                        while iteration < max_iterations:
                            objectives.append(self.objective)
                            solution_changed = False
                            move_generator = self.generate_moves(random_move_order=random_move_order, random_index_order=random_index_order, order=move_order)
                            
                            if run_in_multiprocessing: #If using multiprocessing
                                # Start by updating shared memory arrays
                                np.copyto(shared_closest_distances_intra, self.closest_distances_intra)
                                np.copyto(shared_closest_points_intra, self.closest_points_intra)
                                np.copyto(shared_closest_distances_inter, self.closest_distances_inter)
                                np.copyto(shared_closest_points_inter, self.closest_points_inter_array)
                            
                                event.clear() #reset event for current iteration
                                results = [] #resest results for current iteration
                                current_iteration_time = time.time() #This is for logging purposes as well as tracking for hybrid mode

                                # Try moves in batches
                                while True:
                                    batches = []
                                    for _ in range(max_batches): #fill list with up to max_batches batches
                                        batch = [] #batch of moves
                                        try:
                                            for _ in range(batch_size):
                                                move_type, move_content = next(move_generator)
                                                batch.append((move_type, move_content))
                                        except StopIteration:
                                            if len(batch) > 0:
                                                batches.append(batch)
                                            break
                                        if len(batch) > 0:
                                            batches.append(batch)

                                    # Process current collection of batches in parallel
                                    if len(batches) > 0:
                                        batch_results = []
                                        for batch in batches:
                                            if event.is_set():
                                                break
                                            res = pool.apply_async(
                                                process_batch,
                                                args=(
                                                    batch, event, 
                                                    self.selection_per_cluster, self.nonselection_per_cluster,
                                                    self.objective
                                                ),
                                                callback = lambda result: process_batch_result(result, results)
                                            )
                                            batch_results.append(res)

                                        for result in batch_results:
                                            result.wait()

                                        if len(results) > 0: #if improvement is found, stop processing batches
                                            solution_changed = True
                                            break

                                    else: # No more tasks to process, break while loop
                                        break

                                time_per_iteration.append(time.time() - current_iteration_time)
                                if solution_changed: # If improvement is found, update solution
                                    move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters = results[0]
                                    self.accept_move(move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters)
                                    iteration += 1 #update iteration count
                                else:
                                    break
                            else: # If not using multiprocessing, only in hybrid mode
                                current_iteration_time = time.time() #This is for logging purposes as well as tracking for hybrid mode

                                for move_type, move_content in move_generator:
                                    if move_type == "add":
                                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(move_content, local_search=True)
                                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                                            solution_changed = True
                                            break
                                    elif move_type == "swap":
                                        idx_to_add, idx_to_remove = move_content
                                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                                            solution_changed = True
                                            break
                                    elif move_type == "doubleswap":
                                        idxs_to_add, idx_to_remove = move_content
                                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_doubleswap(idxs_to_add, idx_to_remove)
                                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                                            solution_changed = True
                                            break
                                    elif move_type == "remove":
                                        idx_to_remove = move_content
                                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove, local_search=True)
                                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                                            solution_changed = True
                                            break

                                current_iteration_time = time.time() - current_iteration_time
                                time_per_iteration.append(current_iteration_time)
                                # Check if we need to switch to multiprocessing
                                if current_iteration_time > runtime_switch:
                                    times_runtime_exceeded += 1
                                else:
                                    times_runtime_exceeded = 0
                                if times_runtime_exceeded > num_switch:
                                    run_in_multiprocessing = True
                                    switch_iteration = iteration
                                    if logging:
                                        print(f"Switching to multiprocessing at iteration {iteration} (last {num_switch} runtimes: {time_per_iteration[-num_switch:]})", flush=True)
                            
                                if solution_changed: # If improvement is found, update solution
                                    self.accept_move(move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters)
                                    iteration += 1
                                else:
                                    break
                                
                            if iteration % logging_frequency == 0 and logging:
                                print(f"Iteration {iteration}: Objective = {self.objective:.6f}", flush=True)
                                print(f"Average runtime last {logging_frequency} iterations: {np.mean(time_per_iteration[-logging_frequency:]):.6f} seconds", flush=True)
            except Exception as e:
                print(f"An error occurred during local search: {e}", flush=True)
                raise e
            finally:
                # Clean up shared memory if it was created
                if distances_shm:
                    distances_shm.close()
                    distances_shm.unlink()
                if clusters_shm:
                    clusters_shm.close()
                    clusters_shm.unlink()
                if closest_distances_intra_shm:
                    closest_distances_intra_shm.close()
                    closest_distances_intra_shm.unlink()
                if closest_points_intra_shm:
                    closest_points_intra_shm.close()
                    closest_points_intra_shm.unlink()
                if closest_distances_inter_shm:
                    closest_distances_inter_shm.close()
                    closest_distances_inter_shm.unlink()
                if closest_points_inter_shm:
                    closest_points_inter_shm.close()
                    closest_points_inter_shm.unlink()
        # Single core processing
        else:
            while iteration < max_iterations:
                objectives.append(self.objective)
                solution_changed = False
                move_generator = self.generate_moves(random_move_order=random_move_order, random_index_order=random_index_order)

                current_iteration_time = time.time() #This is for logging purposes as well as tracking for hybrid mode
                for move_type, move_content in move_generator:
                    if move_type == "add":
                        idx_to_add = move_content
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add, local_search=True)
                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                            solution_changed = True
                            break
                    elif move_type == "swap":
                        idx_to_add, idx_to_remove = move_content
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                            solution_changed = True
                            break
                    elif move_type == "doubleswap":
                        idxs_to_add, idx_to_remove = move_content
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_doubleswap(idxs_to_add, idx_to_remove)
                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                            solution_changed = True
                            break
                    elif move_type == "remove":
                        idx_to_remove = move_content
                        candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove, local_search=True)
                        if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                            solution_changed = True
                            break
                
                time_per_iteration.append(time.time() - current_iteration_time)
                if solution_changed: # If improvement is found, update solution
                    self.accept_move(move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters)
                    iteration += 1
                else:
                    break

                if iteration % logging_frequency == 0 and logging:
                    print(f"Iteration {iteration}: Objective = {self.objective:.6f}", flush=True)
                    print(f"Average runtime last {logging_frequency} iterations: {np.mean(time_per_iteration[-logging_frequency:]):.6f} seconds", flush=True)

        return time_per_iteration, objectives, switch_iteration

    def local_search_adaptive(self, max_iterations: int = 10_000, num_cores: int = 2,
                           random_move_order: bool = True, random_index_order: bool = True, move_order: list = ["add", "swap", "doubleswap", "remove"],
                           batch_size: int = 1000, max_batches: int = 32, 
                           runtime_switch: float = 10.0,
                           logging: bool = False, logging_frequency: int = 500,
                           ):
        """
        Perform local search to find a (local) optimal solution using an adaptive approach where
        the search switches between single-core and multi-core execution based on the runtime of iterations.

        Parameters:
        max_iterations: int
            The maximum number of iterations to perform.
        num_cores: int
            The number of cores to use for parallel processing.
            NOTE: if set to 1, local search will always run in
            single processor mode, even if hybrid is True.
        hybrid: bool
            If True, local search will switch to multiprocessing
            after num_switch consecutive iterations take longer
            than runtime_switch seconds.
        random_move_order: bool
            If True, the order of moves (add, swap, doubleswap,
            remove) is randomized.
        random_index_order: bool
            If True, the order of indices for moves is randomized.
            NOTE: if random_move_order is True, but this is false,
            all moves of a particular type will be tried before
            moving to the next move type, but the order of moves
            is random).
        move_order: list
            If provided, this list will be used to determine the
            order of moves. If random_move_order is True, this
            list will be shuffled before use.
            NOTE: this list should contain the following move types (as strings):
                - "add"
                - "swap"
                - "doubleswap"
                - "remove"
            NOTE: by leaving out a move type, it will not be
            considered in the local search.
        batch_size: int
            In multiprocessing mode, moves are processed in batches
            of this size.
            NOTE: do not set this to a value smaller than 0
        max_batches: int
            To prevent memory issues, the number of batches is
            limited to this value. Once every batch has been
            processed, the next set of batches will be
            processed.
            NOTE: this should be set to at least the number of
            num_cores, otherwise some cores will be idle.
        runtime_switch: float
            Threshold in seconds for switching between single-core and multi-core 
            execution.
        logging: bool
            If True, information about the local search will be printed.
        logging_frequency: int
            If logging is True, information will be printed every
            logging_frequency iterations.
        
        Returns:
        --------
        time_per_iteration: list of floats
            The time taken for each iteration.
            NOTE: this is primarily for logging purposes
        objectives: list of floats
            The objective value in each iteration.
        """
        # Validate input parameters
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(num_cores, int) or num_cores < 2:
            raise ValueError("num_cores must be a positive integer and larger than 1.")
        if not isinstance(random_move_order, bool):
            raise ValueError("random_move_order must be a boolean value.")
        if not isinstance(random_index_order, bool):
            raise ValueError("random_index_order must be a boolean value.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(max_batches, int) or max_batches < 1:
            raise ValueError("max_batches must be a positive integer.")
        if not isinstance(runtime_switch, (int, float)) or runtime_switch < 0:
            raise ValueError("runtime_switch must be a non-negative number.")
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")

        # Initialize variables
        iteration = 0
        time_per_iteration = []
        objectives = []
        solution_changed = False
        run_in_multiprocessing = False

        # Multiprocessing
        try:
            # Copy distance matrix to shared memory
            distances_shm = shm.SharedMemory(create=True, size=self.distances.nbytes)
            shared_distances = np.ndarray(self.distances.shape, dtype=self.distances.dtype, buffer=distances_shm.buf)
            np.copyto(shared_distances, self.distances) #this array is static, only copy once
            # Copy cluster assignment to shared memory
            clusters_shm = shm.SharedMemory(create=True, size=self.clusters.nbytes)
            shared_clusters = np.ndarray(self.clusters.shape, dtype=self.clusters.dtype, buffer=clusters_shm.buf)
            np.copyto(shared_clusters, self.clusters) #this array is static, only copy once

            # for the intra and inter distances, only copy them during iterations since they are updated during the local search
            # Copy closest_distances_intra to shared memory
            closest_distances_intra_shm = shm.SharedMemory(create=True, size=self.closest_distances_intra.nbytes)
            shared_closest_distances_intra = np.ndarray(self.closest_distances_intra.shape, dtype=self.closest_distances_intra.dtype, buffer=closest_distances_intra_shm.buf)
            # Copy closest_points_intra to shared memory
            closest_points_intra_shm = shm.SharedMemory(create=True, size=self.closest_points_intra.nbytes)
            shared_closest_points_intra = np.ndarray(self.closest_points_intra.shape, dtype=self.closest_points_intra.dtype, buffer=closest_points_intra_shm.buf)

            # Copy closest_distances_inter to shared memory
            closest_distances_inter_shm = shm.SharedMemory(create=True, size=self.closest_distances_inter.nbytes)
            shared_closest_distances_inter = np.ndarray(self.closest_distances_inter.shape, dtype=self.closest_distances_inter.dtype, buffer=closest_distances_inter_shm.buf)
            # Copy closest_points_inter to shared memory
            closest_points_inter_shm = shm.SharedMemory(create=True, size=self.closest_points_inter_array.nbytes)
            shared_closest_points_inter = np.ndarray(self.closest_points_inter_array.shape, dtype=self.closest_points_inter_array.dtype, buffer=closest_points_inter_shm.buf)

            with Manager() as manager:
                event = manager.Event() #this is used to signal when tasks should be stopped
                results = manager.list() #this is used to store an improvement is one is found

                with Pool(
                    processes=num_cores,
                    initializer=init_worker,
                    initargs=(
                        distances_shm.name, shared_distances.shape,
                        clusters_shm.name, shared_clusters.shape,
                        closest_distances_intra_shm.name, shared_closest_distances_intra.shape,
                        closest_points_intra_shm.name, shared_closest_points_intra.shape,
                        closest_distances_inter_shm.name, shared_closest_distances_inter.shape,
                        closest_points_inter_shm.name, shared_closest_points_inter.shape,
                        self.unique_clusters, self.cost_per_cluster, self.num_points
                    ),
                ) as pool:
                    while iteration < max_iterations:
                        objectives.append(self.objective)
                        solution_changed = False
                        run_in_multiprocessing = False 
                        move_generator = self.generate_moves(random_move_order=random_move_order, random_index_order=random_index_order, order=move_order)

                        current_iteration_time = time.time() #This is for logging purposes and for adaptive mode tracking
                        move_counter = 0
                        for move_type, move_content in move_generator:
                            move_counter += 1
                            if move_type == "add":
                                candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(move_content, local_search=True)
                                if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                                    solution_changed = True
                                    break
                            elif move_type == "swap":
                                idx_to_add, idx_to_remove = move_content
                                candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idx_to_add, idx_to_remove)
                                if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                                    solution_changed = True
                                    break
                            elif move_type == "doubleswap":
                                idxs_to_add, idx_to_remove = move_content
                                candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_doubleswap(idxs_to_add, idx_to_remove)
                                if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                                    solution_changed = True
                                    break
                            elif move_type == "remove":
                                idx_to_remove = move_content
                                candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove, local_search=True)
                                if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                                    solution_changed = True
                                    break
                            if move_counter % 1_000: #every 1000 moves, check if we should switch to multiprocessing
                                if time.time() - current_iteration_time > runtime_switch:
                                    if logging:
                                        print(f"Iteration {iteration+1} is taking longer than {runtime_switch} seconds, switching to multiprocessing.", flush=True)
                                    run_in_multiprocessing = True
                                    break #break out of singleprocessing
                            
                        if run_in_multiprocessing: #If switching to multiprocessing
                            # Start by updating shared memory arrays
                            np.copyto(shared_closest_distances_intra, self.closest_distances_intra)
                            np.copyto(shared_closest_points_intra, self.closest_points_intra)
                            np.copyto(shared_closest_distances_inter, self.closest_distances_inter)
                            np.copyto(shared_closest_points_inter, self.closest_points_inter_array)
                            
                            event.clear() #reset event for current iteration
                            results = [] #resets results for current iteration

                            num_solutions_tried = 0
                            # Try moves in batches
                            while True:
                                batches = []
                                num_this_loop = 0
                                cur_batch_time = time.time()
                                for _ in range(max_batches): #fill list with up to max_batches batches
                                    batch = [] #batch of moves
                                    try:
                                        for _ in range(batch_size):
                                            move_type, move_content = next(move_generator)
                                            batch.append((move_type, move_content))
                                    except StopIteration:
                                        if len(batch) > 0:
                                            batches.append(batch)
                                            num_this_loop += len(batch)
                                        break
                                    if len(batch) > 0:
                                        batches.append(batch)
                                        num_this_loop += len(batch)

                                # Process current collection of batches in parallel
                                if len(batches) > 0:
                                    batch_results = []
                                    for batch in batches:
                                        if event.is_set():
                                            break
                                        res = pool.apply_async(
                                            process_batch,
                                            args=(
                                                batch, event, 
                                                self.selection_per_cluster, self.nonselection_per_cluster,
                                                self.objective
                                            ),
                                            callback = lambda result: process_batch_result(result, results)
                                        )
                                        batch_results.append(res)

                                    for result in batch_results:
                                        result.wait()

                                    if len(results) > 0: #if improvement is found, stop processing batches
                                        solution_changed = True
                                        move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters = results[0]
                                        break
                                    else:
                                        num_solutions_tried += num_this_loop
                                        if logging:
                                            print(f"Processed {num_solutions_tried} solutions (current batch took {time.time() - cur_batch_time:.2f}s), no improvement found yet.", flush=True)

                                else: # No more tasks to process, break while loop
                                    break

                        time_per_iteration.append(time.time() - current_iteration_time)
                        if solution_changed: # If improvement is found, update solution
                            self.accept_move(move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters)
                            iteration += 1 #update iteration count
                        else:
                            break
                                
                        if iteration % logging_frequency == 0 and logging:
                            print(f"Iteration {iteration}: Objective = {self.objective:.6f}", flush=True)
                            print(f"Average runtime last {logging_frequency} iterations: {np.mean(time_per_iteration[-logging_frequency:]):.6f} seconds", flush=True)
        except Exception as e:
            print(f"An error occurred during local search: {e}", flush=True)
            print("Traceback details:", flush=True)
            traceback.print_exc()
            raise e
        finally:
            # Clean up shared memory if it was created
            if distances_shm:
                distances_shm.close()
                distances_shm.unlink()
            if clusters_shm:
                clusters_shm.close()
                clusters_shm.unlink()
            if closest_distances_intra_shm:
                closest_distances_intra_shm.close()
                closest_distances_intra_shm.unlink()
            if closest_points_intra_shm:
                closest_points_intra_shm.close()
                closest_points_intra_shm.unlink()
            if closest_distances_inter_shm:
                closest_distances_inter_shm.close()
                closest_distances_inter_shm.unlink()
            if closest_points_inter_shm:
                closest_points_inter_shm.close()
                closest_points_inter_shm.unlink()

        return time_per_iteration, objectives

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

    def generate_indices_add(self, random: bool = False):
        """
        Generates indices of points that can be added to the solution.
        
        Parameters:
        -----------
        random: bool
            If True, the order of indices is randomized. Default is False.
            NOTE: this uses the random state stored in the Solution object.
        """
        indices = np.flatnonzero(~self.selection)
        if random:
            yield from self.random_state.permutation(indices)
        else:
            yield from indices

    def generate_indices_swap(self, number_to_add: int = 1, random: bool = False):
        """
        Generates indices of pairs of points that can be swapped in the solution.
        NOTE: when running in random mode, we randomly iterate over 
        
        Parameters:
        -----------
        number_to_add: int
            The number of points to add in the swap operation. Default is 1 (remove 1 point, add 1 point).
        random: bool
            If True, the order of indices is randomized. Default is False.
            NOTE: this uses the random state stored in the Solution object.
            NOTE: although the cluster order can be randomized, the cluster is exhausted before moving to the next cluster.
        """
        if random:
            cluster_order = self.random_state.permutation(self.unique_clusters)
        else:
            cluster_order = self.unique_clusters
        for cluster in cluster_order:
            clusters_mask = self.clusters == cluster
            selected = np.where(clusters_mask & self.selection)[0]
            unselected = np.where(clusters_mask & ~self.selection)[0]

            if random:
                if selected.size == 0 or unselected.size == 0: #skip permuting if no points to swap
                    continue
                selected = self.random_state.permutation(selected)
                unselected = self.random_state.permutation(unselected)

            for idx_to_remove in selected:
                if number_to_add == 1:
                    for idx_to_add in unselected:
                        yield idx_to_add, idx_to_remove
                else:
                    for indices_to_add in itertools.combinations(unselected, number_to_add):
                        yield indices_to_add, idx_to_remove

    def generate_indices_remove(self, random=False):
        """
        Generates indices of points that can be removed from the solution.
        
        Parameters:
        -----------
        random: bool
            If True, the order of indices is randomized. Default is False.
            NOTE: This uses the random state stored in the Solution object.
            NOTE: Although the cluster order can be randomized, the cluster is exhausted before moving to the next cluster.
        """
        if random:
            cluster_order = self.random_state.permutation(self.unique_clusters)
        else:
            cluster_order = self.unique_clusters
        for cluster in cluster_order:
            if len(self.selection_per_cluster[cluster]) > 1:
                if random:
                    for idx in self.random_state.permutation(list(self.selection_per_cluster[cluster])):
                        yield idx
                else:
                    for idx in self.selection_per_cluster[cluster]:
                        yield idx

    def generate_moves(self, random_move_order: bool = True, random_index_order: bool = True, order=["add", "swap", "doubleswap", "remove"]):
        """
        Creates a generator that generates moves in a specific order, or
        random order.
        
        Parameters:
        -----------
        random_move_order: bool
            If True, the order of move types (add, swap, doubleswap, remove) is randomized.
        random_index_order: bool
            If True, the order of indices for each move type is randomized.
            NOTE: If random_move_order is False, this will still randomize the order in which
            indices are generated for each move type, but the order of move types will
            be fixed as specified in the 'order' parameter.
        order: list
            The order of move types to generate. This should be a list containing
            the move types as strings: "add", "swap", "doubleswap", "remove".
            NOTE: If random_move_order is False, the order as specified in this list will
            be used.
            NOTE: Moves can be omitted by not including them in this list.
        """
        generators = {}
        # Add move types to generators dictionary
        for move_type in order:
            if move_type == "add":
                generators[move_type] = self.generate_indices_add(random=random_index_order)
            elif move_type == "swap":
                generators[move_type] = self.generate_indices_swap(number_to_add=1, random=random_index_order)
            elif move_type == "doubleswap":
                generators[move_type] = self.generate_indices_swap(number_to_add=2, random=random_index_order)
            elif move_type == "remove":
                generators[move_type] = self.generate_indices_remove(random=random_index_order)
            else:
                raise ValueError(f"Unknown move type: {move_type}")
        active_generators = order.copy()

        # While there are active generators, yield from them until exhausted
        while active_generators:
            if random_move_order:
                selected_generator = self.random_state.choice(active_generators)
            else:
                selected_generator = active_generators[0]
            # This try-except block allows to yield from generator, and if no more of the corresponding move, removes it from active generators
            try:
                yield selected_generator, next(generators[selected_generator])
            except StopIteration:
                active_generators.remove(selected_generator)
    
"""
Here we define helper functions that can be used by the multiprocessing version of the local search.
The key characteristic of these functions is that they do not rely on an explicit instance of the
Solution class, but rather use shared memory as well as initialized variables to evaluate moves
in parallel.
"""
def evaluate_add_mp(
        idx_to_add: int, objective: float,
        selection_per_cluster: dict, nonselection: set
        ):
        """
        Evaluates the effect of adding an unselected point to the solution.
        NOTE: this function relies on shared memory, as well as existing variables that
        have to be initialized (those starting with an underscore) when spawning a worker 
        process!
        NOTE: in the current implementation, there is no check for feasibility, so it is assumed
        that the point can be added without violating any constraints!

        Parameters:
        -----------
        idx_to_add: int
            The index of the point to be added.
        objective: float
            The current objective value of the solution.
        selection_per_cluster: dict
            A dictionary mapping cluster indices to sets of selected point indices in that cluster.
        nonselection: set
            A set of indices of points (in the cluster of the point to be added) that are currently 
            not selected in the solution.
        """
        cluster = _clusters[idx_to_add]
        candidate_objective = objective + _cost_per_cluster[cluster] # cost for adding the point
        
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
        if candidate_objective > objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
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

        if candidate_objective < objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
            return candidate_objective, add_within_cluster, add_for_other_clusters
        else:
            return -1, -1, -1  # -1, -1, -1 to signify no improvement

def evaluate_swap_mp(
        idxs_to_add, idx_to_remove: int, objective: float,
        selection_per_cluster: dict, nonselection: set):
    """
    Evaluates the effect of swapping a selected point for an unselected point (or multiple)
    in the solution.
    NOTE: this function relies on shared memory, as well as existing variables that
    have to be initialized (those starting with an underscore) when spawning a worker
    process!
    NOTE: in the current implementation, there is no check for feasibility, so it is assumed
    that the swap can be performed without violating any constraints!
    
    Parameters:
    -----------
    idxs_to_add: int or list of int
        The index or indices of the point(s) to be added.
    idx_to_remove: int
        The index of the point to be removed.
    objective: float
        The current objective value of the solution.
    selection_per_cluster: dict
        A dictionary mapping cluster indices to sets of selected point indices in that cluster.
    nonselection: set
        A set of indices of points (in the cluster of the point to be removed) that are currently 
        not selected in the solution.
    """
    try:
        num_to_add = len(idxs_to_add)
    except TypeError:
        num_to_add = 1
        idxs_to_add = [idxs_to_add]
    cluster = _clusters[idx_to_remove]
    candidate_objective = objective + (num_to_add - 1) * _cost_per_cluster[cluster] # cost for adding and removing
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
            cur_closest_point = _closest_points_inter[other_cluster, cluster]
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
    if candidate_objective < objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
        return candidate_objective, add_within_cluster, add_for_other_clusters
    else:
        return -1, -1, -1  # -1, -1, -1 to signify no improvement

def evaluate_remove_mp(
        idx_to_remove: int, objective: float,
        selection_per_cluster: dict, nonselection: set,
        ):
    """
    Evaluates the effect of removing a selected point from the solution.
    NOTE: this function relies on shared memory, as well as existing variables that
    have to be initialized (those starting with an underscore) when spawning a worker
    process!
    NOTE: in the current implementation, there is no check for feasibility, so it is assumed
    that the swap can be performed without violating any constraints!

    Parameters:
    -----------
    idx_to_remove: int
        The index of the point to be removed.
    objective: float
        The current objective value of the solution.
    selection_per_cluster: dict
        A dictionary mapping cluster indices to sets of selected point indices in that cluster.
    nonselection: set
        A set of indices of points (in the cluster of the point to be removed) that are currently 
        not selected in the solution.
    """
    cluster = _clusters[idx_to_remove]
    candidate_objective = objective - _cost_per_cluster[cluster] # cost for removing the point from the cluster
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
            cur_closest_point = _closest_points_inter[cluster, other_cluster]
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
    
    if candidate_objective > objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
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

    if candidate_objective < objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
        return candidate_objective, add_within_cluster, add_for_other_clusters
    else:
        return -1, -1, -1

def init_worker(
        distances_name, distances_shape, 
        clusters_name, clusters_shape,
        closest_distances_intra_name, closest_distances_intra_shape, 
        closest_points_intra_name, closest_points_intra_shape,
        closest_distances_inter_name, closest_distances_inter_shape,
        closest_points_inter_name, closest_points_inter_shape,
        unique_clusters, cost_per_cluster, num_points):
    """
    Initializes a worker for multiprocessing by setting up shared memory
    for the distances, clusters, closest distances and points.

    Parameters:
    -----------
    distances_name: str
        Name of the shared memory segment for distances.
    distances_shape: tuple
        Shape of the distances array.
    clusters_name: str
        Name of the shared memory segment for clusters.
    clusters_shape: tuple
        Shape of the clusters array.
    closest_distances_intra_name: str
        Name of the shared memory segment for intra-cluster closest distances.
    closest_distances_intra_shape: tuple
        Shape of the intra-cluster closest distances array.
    closest_points_intra_name: str
        Name of the shared memory segment for intra-cluster closest points.
    closest_points_intra_shape: tuple
        Shape of the intra-cluster closest points array.
    closest_distances_inter_name: str
        Name of the shared memory segment for inter-cluster closest distances.
    closest_distances_inter_shape: tuple
        Shape of the inter-cluster closest distances array.
    closest_points_inter_name: str
        Name of the shared memory segment for inter-cluster closest points.
    closest_points_inter_shape: tuple
        Shape of the inter-cluster closest points array.
    unique_clusters: np.ndarray
        Array of unique cluster indices.
    cost_per_cluster: np.ndarray
        Costs associated with selecting a point from a cluster.
        NOTE: This cost may be different for each cluster.
    num_points: int
        Total number of points in the dataset.
    """
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
    global _closest_points_inter_shm, _closest_points_inter
    _closest_points_inter_shm = shm.SharedMemory(name=closest_points_inter_name)
    _closest_points_inter = np.ndarray(closest_points_inter_shape, dtype=np.int32, buffer=_closest_points_inter_shm.buf)
    global _unique_clusters, _cost_per_cluster, _num_points
    _unique_clusters = unique_clusters
    _cost_per_cluster = cost_per_cluster
    _num_points = num_points

    # Define clean up function to close shared memory
    def cleanup():
        try:
            _distances_shm.close()
            _clusters_shm.close()
            _closest_distances_intra_shm.close()
            _closest_points_intra_shm.close()
            _closest_distances_inter_shm.close()
            _closest_points_inter_shm.close()
        except Exception as e:
            print(f"Error closing shared memory: {e}")

    atexit.register(cleanup)

def process_batch(batch, event, selection_per_cluster, nonselection_per_cluster, objective):
    """
    Processes a batch of tasks (used with multiprocessing).

    Parameters:
    -----------
    batch: list of tuples
        Each tuple contains a task type and its content.
        Task types can be "add", "swap", "doubleswap", or "remove".
    event: multiprocessing.Event
        An event to signal when a solution improvement is found.
    selection_per_cluster: dict
        A dictionary mapping cluster indices to sets of selected points in that cluster.
    nonselection_per_cluster: dict
        A dictionary mapping cluster indices to sets of non-selected points in that cluster.
    objective: float
        The current objective value of the solution.

    Returns:
    --------
    tuple
        A tuple containing the move type, move content, candidate objective,
        add_within_cluster, and add_for_other_clusters if an improvement is found,
        otherwise (None, None, -1, None, None).
    """
    global _distances, _clusters, _closest_distances_intra, _closest_points_intra, _closest_distances_inter, _closest_points_inter
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
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_swap_mp(idxs_to_add, idx_to_remove, objective, selection_per_cluster, nonselection_per_cluster[cluster])
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 1
                event.set()
                return task, content, candidate_objective, add_within_cluster, add_for_other_clusters
        elif task == "remove":
            idx_to_remove = content
            cluster = _clusters[idx_to_remove]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_remove_mp(idx_to_remove, objective, selection_per_cluster, nonselection_per_cluster[cluster])
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 1
                event.set()
                return "remove", content, candidate_objective, add_within_cluster, add_for_other_clusters

    return None, None, -1, None, None

def process_batch_result(result, results_list):
    """
    Adds the result of a move evaluation to the results list if the candidate objective is
    an improvement (otherwise it is ignored).
    NOTE: this modifies the results_list in place.

    Parameters:
    -----------
    result: tuple
        A tuple containing the move type, move content, candidate objective,
        add_within_cluster, and add_for_other_clusters.
    results_list: list
        A list to which the result will be added if the candidate objective is an improvement.
    """
    move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters = result
    if candidate_objective > -1:
        results_list.append((move_type, move_content, candidate_objective, add_within_cluster, add_for_other_clusters))

def get_index(idx1: int, idx2: int, num_points: int):
    """
    Returns the index in the condensed distance matrix for the given pair of indices.

    Parameters:
    -----------
    idx1: int
        Index of the first point.
    idx2: int
        Index of the second point.
    num_points: int
        Total number of points in the dataset.

    Returns:
    --------
    int
        The index in the condensed distance matrix for the given pair of indices.
    """
    if idx1 == idx2:
        return -1
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    return num_points * idx1 - (idx1 * (idx1 + 1)) // 2 + idx2 - idx1 - 1

def get_distance(idx1: int, idx2: int, distances: np.ndarray, num_points: int):
    """
    Returns the distance between two points which has to be
    converted since the distance matrix is stored as a
    condensed matrix.

    Parameters:
    -----------
    idx1: int
        Index of the first point.
    idx2: int
        Index of the second point.
    distances: np.ndarray
        Condensed distance matrix.
    num_points: int
        Total number of points in the dataset.
        
    Returns:
    --------
    float
        The distance between the two points.
    """
    if idx1 == idx2:
        return 0.0
    index = get_index(idx1, idx2, num_points)
    return distances[index]

if __name__ == "__main__":
    pass