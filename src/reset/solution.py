"""
TO-DO:
    - Potential problem with cluster labels that are not necessarily 0-indexed or consecutive!!
"""
import numpy as np
from scipy.spatial.distance import squareform
import itertools
import math
import multiprocessing.shared_memory as shm
import multiprocessing as mp
from queue import Empty
import time
import traceback

# This is to define the precision threshold for floating point comparisons
PRECISION_THRESHOLD = 1e-10
DISTANCE_DTYPE = np.float32
AUXILIARY_DISTANCE_DTYPE = np.float64

# This encodes move types for multiprocessing local search
MOVE_ADD = 0
MOVE_SWAP = 1
MOVE_DSWAP = 2
MOVE_REMOVE = 3
MOVE_SYNC = 99
MOVE_STOP = 100

# Global variable for multiprocessing worker solutions (i.e. attached copies)
_WORKER_SOL = None

# Single processing solution class (stable)
class Solution:
    def __init__(self, distances: np.ndarray, clusters: np.ndarray, selection: np.ndarray = None, selection_cost: float = 1.0, cost_per_cluster: int = 0, seed=None):
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
        self.distances = squareform(distances.astype(dtype=DISTANCE_DTYPE))
        self.clusters = clusters.astype(dtype=np.int64)
        self.unique_clusters = np.unique(self.clusters)
        # Cost per cluster based on number of points in each cluster
        # If cost_per_cluster is True, then the cost is divided by the number of points in each cluster
        # cost_per_cluster is indexed by cluster indices
        self.selection_cost = selection_cost
        self.cost_per_cluster = np.zeros(self.unique_clusters.shape[0], dtype=AUXILIARY_DISTANCE_DTYPE)
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
            elif cost_per_cluster == -2:
                # Define the average distance in a cluster as the average similarity
                # of all points in the cluster to the centroid of the cluster.
                cluster_points = np.where(self.clusters == cluster)[0]
                centroid = np.argmin(np.sum(distances[np.ix_(cluster_points, cluster_points)], axis=1))
                self.cost_per_cluster[cluster] = selection_cost * ( 1.0-np.mean(distances[centroid, cluster_points]) )
            elif cost_per_cluster == 3:
                # Define the average distance in a cluster as the average distance
                # of all points in the cluster to the closest point in the cluster.
                cluster_points = np.where(self.clusters == cluster)[0]
                self.cost_per_cluster[cluster] = np.mean([np.min(distances[point, cluster_points]) for point in cluster_points])
        self.num_points = distances.shape[0]

        # Process initial representation to optimize for comparisons speed
        self.points_per_cluster = {cluster: set(np.where(self.clusters == cluster)[0]) for cluster in self.unique_clusters} #points in every cluster

        # Calculate objective
        self.calculate_objective()
        
    @classmethod
    def generate_centroid_solution(cls, distances, clusters, selection_cost: float = 1.0, cost_per_cluster: int = 0, seed=None):
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

        return cls(distances, clusters, selection=selection, selection_cost=selection_cost, cost_per_cluster=cost_per_cluster, seed=seed)
    
    @classmethod
    def generate_random_solution(cls, distances, clusters, selection_cost: float = 1.0, cost_per_cluster: int = 0, max_fraction=0.1, seed=None):
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

        return cls(distances, clusters, selection, selection_cost=selection_cost, cost_per_cluster=cost_per_cluster, seed=random_state)

    # Core state and feasibility methods
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
        self.closest_distances_intra = np.zeros(self.selection.shape[0], dtype=AUXILIARY_DISTANCE_DTYPE) #distances to closest selected point
        self.closest_points_intra = np.arange(0, self.selection.shape[0], dtype=np.int32) #indices of closest selected point
        # INTER CLUSTER INFORMATION
        self.closest_distances_inter = np.zeros((self.unique_clusters.shape[0], self.unique_clusters.shape[0]), dtype=AUXILIARY_DISTANCE_DTYPE) #distances to closest selected point
        self.closest_points_inter = np.zeros((self.unique_clusters.shape[0], self.unique_clusters.shape[0]), dtype=np.int32) #indices of closest selected point
        """
        Interpretation of closest_points_inter: given a pair of clusters (cluster1, cluster2),
        the value at closest_points_inter[cluster1, cluster2] is the index of the point in cluster1 that is closest to any point in cluster2.
        In principle this thus assumes that the leading index is the "from" cluster and thus yields
        the point in that cluster that is closest to any any point in cluster2 (which can be retrieved from closest_points_inter[cluster2, cluster1]).
        """

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
                cur_min = AUXILIARY_DISTANCE_DTYPE(np.inf)
                cur_idx = None # index of the closest selected point of the same cluster
                for other_idx in sorted(list(self.selection_per_cluster[cluster])): #this is to ensure consistent ordering
                    cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                    if cur_dist < cur_min:
                        cur_min = cur_dist
                        cur_idx = other_idx
                self.closest_distances_intra[idx] = AUXILIARY_DISTANCE_DTYPE(cur_min)
                self.closest_points_intra[idx] = np.int32(cur_idx)
                objective += cur_min
        # Inter cluster distance costs
        for cluster_1, cluster_2 in itertools.combinations(self.unique_clusters, 2):
            cur_max = -np.float64(np.inf)
            cur_pair = (None, None) # indices of the closest selected points of the two clusters
            for point_1 in sorted(list(self.selection_per_cluster[cluster_1])): #this is to ensure consistent ordering
                for point_2 in sorted(list(self.selection_per_cluster[cluster_2])): #this is to ensure consistent ordering
                    cur_dist = 1.0 - get_distance(point_1, point_2, self.distances, self.num_points)
                    if cur_dist > cur_max:
                        cur_max = cur_dist
                        cur_pair = (point_1, point_2)
            self.closest_distances_inter[cluster_1, cluster_2] = cur_max
            self.closest_distances_inter[cluster_2, cluster_1] = cur_max
            self.closest_points_inter[cluster_1, cluster_2] = cur_pair[0]
            self.closest_points_inter[cluster_2, cluster_1] = cur_pair[1]
            objective += cur_max
        self.objective = objective

    # Local search evaluation and acceptance methods
    def evaluate_add(self, idx_to_add: int, local_search=False):
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
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
            NOTE: new_closest_point will always be idx_to_add.
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, (point_in_this_cluster, point_in_other_cluster), new_distance)]
            NOTE: point_in_this_cluster will always be idx_to_add.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        if self.selection[idx_to_add]:
            raise ValueError("The point to add must not be selected.")
        cluster = self.clusters[idx_to_add]

        # Calculate selection cost
        candidate_objective = self.objective + self.cost_per_cluster[cluster] # cost for adding the point

        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for idx in self.nonselection_per_cluster[cluster]:
            cur_dist = get_distance(idx, idx_to_add, self.distances, self.num_points) # distance to current point (idx)
            if cur_dist < self.closest_distances_intra[idx]:
                candidate_objective += cur_dist - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, idx_to_add, cur_dist))

        # NOTE: Inter-cluster distances can only increase when adding a point, so when doing local search we can exit here if objective is worse
        if candidate_objective > self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD and local_search:
            return np.inf, None, None

        # Calculate inter-cluster distances for other clusters
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
                    add_for_other_clusters.append((other_cluster, (idx_to_add, cur_idx), cur_max))

        return candidate_objective, add_within_cluster, add_for_other_clusters

    def evaluate_swap(self, idxs_to_add, idx_to_remove: int):
        """
        Evaluates the effect of swapping a selected point for a/multiple unselected point(s)
        in the solution.

        Parameters:
        -----------
        idxs_to_add: tuple of int or list of int
            The index or indices of the point(s) to be added.
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
            Structure: [(index_other_cluster, (point_in_this_cluster, point_in_other_cluster), new_distance)]
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        try:
            num_to_add = len(idxs_to_add)
        except TypeError: #assumption is that this is an int
            num_to_add = 1
            idxs_to_add = [idxs_to_add]
        for idx in idxs_to_add:
            if self.selection[idx]:
                raise ValueError("The points to add must not be selected.")
        if not self.selection[idx_to_remove]:
            raise ValueError("The point to remove must be selected.")
        cluster = self.clusters[idx_to_remove]
        for idx in idxs_to_add:
            if self.clusters[idx] != cluster:
                raise ValueError("All points must be in the same cluster.")
            
        # Generate pool of alternative points to compare to
        new_selection = set(self.selection_per_cluster[cluster])
        for idx in idxs_to_add:
            new_selection.add(idx)
        new_selection.remove(idx_to_remove)
        new_nonselection = set(self.nonselection_per_cluster[cluster])
        new_nonselection.add(idx_to_remove)

        # Calculate selection cost
        candidate_objective = self.objective + (num_to_add - 1) * self.cost_per_cluster[cluster] #cost for swapping points

        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
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
            else: #point to be removed is not closest, check if one of newly added points is closer
                cur_dists = [(get_distance(idx, idx_to_add, self.distances, self.num_points), idx_to_add) for idx_to_add in idxs_to_add]
                cur_dist, idx_to_add = min(cur_dists, key=lambda x: x[0])
                if cur_dist < cur_closest_distance:
                    candidate_objective += cur_dist - cur_closest_distance
                    add_within_cluster.append((idx, idx_to_add, cur_dist))

        # Calculate inter-cluster distances for all other clusters
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                cur_closest_similarity = self.closest_distances_inter[cluster, other_cluster]
                cur_closest_point = self.closest_points_inter[cluster, other_cluster]
                cur_closest_pair = (-1, -1)
                if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                    cur_closest_similarity = -np.inf
                    for idx in self.selection_per_cluster[other_cluster]:
                        for other_idx in new_selection:
                            cur_similarity = 1.0 - get_distance(idx, other_idx, self.distances, self.num_points)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                cur_closest_pair = (other_idx, idx)
                else: #point to be removed is not closest, check if one of newly added points is closer
                    for idx in self.selection_per_cluster[other_cluster]:
                        cur_similarities = [(1.0 - get_distance(idx, idx_to_add, self.distances, self.num_points), idx_to_add) for idx_to_add in idxs_to_add]
                        cur_similarity, idx_to_add = max(cur_similarities, key = lambda x: x[0])
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            cur_closest_pair = (idx_to_add, idx)
                if cur_closest_pair[0] > -1:
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))

        return candidate_objective, add_within_cluster, add_for_other_clusters

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
            If False, it will always evaluate the removal.
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the removal.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, (point_in_this_cluster, point_in_other_cluster), new_distance)]
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate removal.")
        if not self.selection[idx_to_remove]:
            raise ValueError("The point to remove must be selected.")
        cluster = self.clusters[idx_to_remove]

        # Generate pool of alternative points to compare to
        new_selection = set(self.selection_per_cluster[cluster])
        new_selection.remove(idx_to_remove)
        new_nonselection = set(self.nonselection_per_cluster[cluster])
        new_nonselection.add(idx_to_remove)

        # Calculate selection cost
        candidate_objective = self.objective - self.cost_per_cluster[cluster]

        # Calculate inter-cluster distances for all other clusters
        # NOTE: Intra-cluster distances can only increase when removing a point, Thus if inter-cluster distances
        # increase, we can exit early.
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                cur_closest_similarity = self.closest_distances_inter[cluster, other_cluster]
                cur_closest_point = self.closest_points_inter[cluster, other_cluster]
                cur_closest_pair = (-1, -1)
                if cur_closest_point == idx_to_remove:
                    cur_closest_similarity = -np.inf
                    for idx in self.selection_per_cluster[other_cluster]:
                        for other_idx in new_selection:
                            cur_similarity = 1.0 - get_distance(idx, other_idx, self.distances, self.num_points)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
        
        # NOTE: Intra-cluster distances can only increase when removing a point, so when doing local search we can exit here if objective is worse
        if candidate_objective > self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD and local_search:
            return np.inf, None, None
        
        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for idx in new_nonselection:
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

    def accept_move(self, idxs_to_add: list, idxs_to_remove: list, candidate_objective: float, add_within_cluster: list, add_for_other_clusters: list):
        """
        Accepts a move to the solution, where multiple points can be added and removed at once.
        NOTE: This assumes that the initial solution and the move
        are feasible and will not check for this.

        PARAMETERS:
        -----------
        idxs_to_add: list of int
            The indices of the points to be added.
            NOTE: This assumes that all indices to be added are in the same cluster (which should be the same as the indices to remove)!
        idxs_to_remove: list of int
            The indices of the points to be removed.
            NOTE: This assumes that all indices to be removed are in the same cluster (which should be the same as the indices to add)!
        candidate_objective: float
            The objective value of the solution after the move.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, (closest_point_this_cluster, closest_point_other_cluster), new_distance)]
        """
        found_clusters = set()
        for idx in idxs_to_add + idxs_to_remove:
            found_clusters.add(self.clusters[idx])
        if len(found_clusters) != 1:
            raise ValueError("All points to add and remove must be in the same cluster.")
        cluster = found_clusters.pop()
        # Updating state attributes of this solution object
        for idx_to_add in idxs_to_add:
            self.selection[idx_to_add] = True
            self.selection_per_cluster[cluster].add(idx_to_add)
            self.nonselection_per_cluster[cluster].remove(idx_to_add)
        for idx_to_remove in idxs_to_remove:
            self.selection[idx_to_remove] = False
            self.selection_per_cluster[cluster].remove(idx_to_remove)
            self.nonselection_per_cluster[cluster].add(idx_to_remove)
        # Updating intra-cluster distances and points
        for idx_to_change, new_closest_point, new_distance in add_within_cluster:
            self.closest_distances_intra[idx_to_change] = new_distance
            self.closest_points_intra[idx_to_change] = new_closest_point
        # Updating inter-cluster distances and points
        for other_cluster, (closest_point_this_cluster, closest_point_other_cluster), new_distance in add_for_other_clusters:
            self.closest_distances_inter[cluster, other_cluster] = new_distance
            self.closest_distances_inter[other_cluster, cluster] = new_distance
            self.closest_points_inter[cluster, other_cluster] = closest_point_this_cluster
            self.closest_points_inter[other_cluster, cluster] = closest_point_other_cluster

        self.objective = candidate_objective

    # Local search move generation methods
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

    def generate_indices_swap_old(self, number_to_add: int = 1, random: bool = False):
        """
        Generates indices of pairs of points that can be swapped in the solution.
        NOTE: when running in random mode, we randomly iterate over 
        NOTE: THIS VERSION IS DEPRECATED, USE generate_indices_swap INSTEAD!
        
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
                        yield [idx_to_add], idx_to_remove
                else:
                    for indices_to_add in itertools.combinations(unselected, number_to_add):
                        yield list(indices_to_add), idx_to_remove

    def generate_indices_swap(self, number_to_add: int = 1, random: bool = False):
        """
        Creates a generator for every cluster, so that
        clusters can be exhausted in random order (opposed to exhausting one cluster at a time).

        Parameters:
        -----------
        number_to_add: int
            The number of points to add in the swap operation. Default is 1 (remove 1 point, add 1 point).
        random: bool
            If True, the order of indices is randomized. Default is False.
            NOTE: this uses the random state stored in the Solution object.
            NOTE: although the cluster order can be randomized, it exhausts all swaps for a given remove index
            in randomized order.

        Yields:
        -------
        idxs_to_add: list of int
            Indices of points to add to the solution.
        idx_to_remove: int
            Index of point to remove from the solution.
        """ 
        cluster_iterators = {}
        for cluster in self.unique_clusters:
            cluster_iterators[cluster] = self._generate_swaps_in_cluster(cluster, number_to_add, random)

        remaining_clusters = list(cluster_iterators.keys())
        while remaining_clusters:
            # With random, randomly select a cluster to yield from
            if random:
                try:
                    cur_cluster = self.random_state.choice(remaining_clusters)
                    yield next( cluster_iterators[cur_cluster] )
                except StopIteration:
                    cluster_iterators.pop(cur_cluster)
                    remaining_clusters.remove(cur_cluster)
            # In non-random, just go through clusters in order
            else:
                cur_cluster = remaining_clusters[0]
                while True:
                    try:
                        yield next( cluster_iterators[cur_cluster] )
                    except StopIteration:
                        cluster_iterators.pop(cur_cluster)
                        remaining_clusters.remove(cur_cluster)
                        break

    def _generate_swaps_in_cluster(self, cluster: int, number_to_add: int = 1, random: bool = False):
        """
        Helper function to generate swaps within a cluster.

        Parameters:
        -----------
        cluster: int
            The cluster to generate swaps for.
        number_to_add: int
            The number of points to add in the swap operation.
        random: bool
            If True, the order of indices is randomized. Default is False.

        Yields:
        -------
        idxs_to_add: list of int
            Indices of points to add to the solution.
        idx_to_remove: int
            Index of point to remove from the solution.
        """
        selected = list(self.selection_per_cluster[cluster])
        unselected = list(self.nonselection_per_cluster[cluster])

        if len(selected) == 0 or len(unselected) < number_to_add: #skip permuting if no points to swap
            return #empty generator
        
        if random:
            selected = self.random_state.permutation(selected)
            unselected = self.random_state.permutation(unselected)

        for idx_to_remove in selected:
            if number_to_add == 1:
                for idx_to_add in unselected:
                    yield [idx_to_add], idx_to_remove
            else:
                for indices_to_add in itertools.combinations(unselected, number_to_add):
                    yield list(indices_to_add), idx_to_remove

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
            cur_list = list(self.selection_per_cluster[cluster])
            if len(self.selection_per_cluster[cluster]) > 1:
                if random:
                    #for idx in self.random_state.permutation(sorted(list(self.selection_per_cluster[cluster]))):
                    for idx in self.random_state.permutation(cur_list):
                        yield idx
                else:
                    for idx in self.selection_per_cluster[cluster]:
                        yield idx

    # Local search (single processing, for multiprocessing see Solution_shm)
    def local_search(self,
                    max_iterations: int = 10_000, max_runtime: float = np.inf,
                    random_move_order: bool = True, random_index_order: bool = True, move_order: list = ["add", "swap", "doubleswap", "remove"],
                    doubleswap_time_threshold: float = 60.0,
                    logging: bool = False, logging_frequency: int = 500,
                    ):
        """
        Perform local search to find a (local) optimal solution using a single processor. 
        
        Parameters:
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        max_runtime: float
            The maximum runtime in seconds for the local search.
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
        doubleswap_time_threshold: float
            The time threshold in seconds after which doubleswaps will no
            longer be considered in the local search.
            NOTE: this is on a per-iteration basis, so if an iteration
            takes longer than this threshold, doubleswaps will be
            skipped in current iteration, but re-added for the next iteration.
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
        """
        # Validate input parameters
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(random_move_order, bool):
            raise ValueError("random_move_order must be a boolean value.")
        if not isinstance(random_index_order, bool):
            raise ValueError("random_index_order must be a boolean value.")
        if not isinstance(move_order, list):
            raise ValueError("move_order must be a list of move types.")
        else:
            if len(move_order) == 0:
                raise ValueError("move_order must contain at least one move type.")
            valid_moves = {"add", "swap", "doubleswap", "remove"}
            if len(set(move_order) - valid_moves) > 0:
                raise ValueError("move_order must contain only the following move types: add, swap, doubleswap, remove.")
        if not isinstance(doubleswap_time_threshold, (int, float)) or doubleswap_time_threshold <= 0:
            raise ValueError("doubleswap_time_threshold must be a positive number.")
        if not isinstance(logging, bool):
            raise ValueError("logging must be a boolean value.")
        if not isinstance(logging_frequency, int) or logging_frequency < 1:
            raise ValueError("logging_frequency must be a positive integer.")  
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")
        
        # Initialize variables for tracking the local search progress
        iteration = 0
        time_per_iteration = []
        objectives = []
        solution_changed = False

        start_time = time.time()
        while iteration < max_iterations:
            current_iteration_time = time.time()
            objectives.append(self.objective)
            solution_changed = False

            # Create move generators for every movetype so doubleswaps can be removed if needed
            move_generator = {}
            for move_type in move_order:
                if move_type == "add":
                    move_generator["add"] = self.generate_indices_add(random=random_index_order)
                elif move_type == "swap":
                    move_generator["swap"] = self.generate_indices_swap(number_to_add=1, random=random_index_order)
                elif move_type == "doubleswap":
                    move_generator["doubleswap"] = self.generate_indices_swap(number_to_add=2, random=random_index_order)
                elif move_type == "remove":
                    move_generator["remove"] = self.generate_indices_remove(random=random_index_order)
            active_moves = move_order.copy() #list of move types for this iteration
            
            move_counter = 0
            while active_moves:
                # Select next move type
                if random_move_order:
                    selected_generator = self.random_state.choice(active_moves)
                else:
                    selected_generator = active_moves[0]
                # Get next move from generator
                try:
                    move_content = next(move_generator[selected_generator])
                    move_type = selected_generator
                except StopIteration: #clear move from generator if no more moves are available
                    active_moves.remove(selected_generator)
                    del move_generator[selected_generator]
                    continue

                move_counter += 1
                if move_type == "add":
                    idx_to_add = move_content
                    idxs_to_add = [idx_to_add]
                    idxs_to_remove = []
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add, local_search=True)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                        solution_changed = True
                        break
                elif move_type == "swap" or move_type == "doubleswap":
                    idxs_to_add, idx_to_remove = move_content
                    idxs_to_remove = [idx_to_remove]
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idxs_to_add, idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                        solution_changed = True
                        break
                elif move_type == "remove":
                    idxs_to_add = []
                    idx_to_remove = move_content
                    idxs_to_remove = [idx_to_remove]
                    candidate_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove, local_search=True)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                        solution_changed = True
                        break

                if move_counter % 1_000 == 0:
                    # Check if total runtime exceeds max_runtime
                    if time.time() - start_time > max_runtime:
                        if logging:
                            print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time}), stopping local search.", flush=True)
                        return time_per_iteration, objectives
                    # Check if doubleswaps should be removed
                    if time.time() - current_iteration_time > doubleswap_time_threshold and "doubleswap" in active_moves:
                        active_moves.remove("doubleswap")
                        del move_generator["doubleswap"]
                        if logging:
                            print(f"Iteration {iteration}: Removed doubleswap moves due to time threshold exceeded ({time.time() - current_iteration_time:.2f} seconds).", flush=True)
                        

            time_per_iteration.append(time.time() - current_iteration_time)
            if solution_changed: # If improvement is found, update solution
                self.accept_move(idxs_to_add, idxs_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                del idxs_to_add, idxs_to_remove #sanity check, should throw error if something weird happens
                iteration += 1
                # Check if time exceeds allowed runtime
                if time.time() - start_time > max_runtime:
                    if logging:
                        print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time}), stopping local search.", flush=True)
                    return time_per_iteration, objectives
            else:
                break

            if iteration % logging_frequency == 0 and logging:
                print(f"Iteration {iteration}: Objective = {self.objective:.10f}", flush=True)
                print(f"Average runtime last {logging_frequency} iterations: {np.mean(time_per_iteration[-logging_frequency:]):.6f} seconds", flush=True)

        return time_per_iteration, objectives

    # Equality check
    def __eq__(self, other):
        """
        Check if two solutions are equal.
        NOTE: This purely checks if all attributes are equal, excluding the random state.
        """
        # Check if other is an instance of the same class
        if not isinstance(other, type(self)):
            print("Other object is not of the same type as self.")
            return False
        # Check if selections are equal
        try:
            if not np.array_equal(self.selection, other.selection):
                print("Selections are not equal.")
                return False
        except:
            print("Selections could not be compared.")
            return False
        # Check if distances are equal
        try:
            if not np.allclose(self.distances, other.distances, atol=PRECISION_THRESHOLD):
                print("Distances are not equal.")
                return False
        except:
            print("Distances could not be compared.")
            return False
        # Check if clusters are equal
        try:
            if not np.array_equal(self.clusters, other.clusters):
                print("Clusters are not equal.")
                return False
        except:
            print("Clusters could not be compared.")
            return False
        # Check if unique clusters are equal
        try:
            if not np.array_equal(self.unique_clusters, other.unique_clusters):
                print("Unique clusters are not equal.")
                return False
        except:
            print("Unique clusters could not be compared.")
            return False
        # Check if selection cost is equal
        if not math.isclose(self.selection_cost, other.selection_cost, rel_tol=PRECISION_THRESHOLD):
            print("Selection costs are not equal.")
            return False
        # Check if cost per cluster is equal
        try:
            if not np.allclose(self.cost_per_cluster, other.cost_per_cluster, atol=PRECISION_THRESHOLD):
                print("Cost per cluster is not equal.")
                return False
        except:
            print("Cost per cluster could not be compared.")
            return False
        # Check if number of points is equal
        if self.num_points != other.num_points:
            print("Number of points is not equal.")
            return False
        # Check if points per cluster are equal
        if set(self.points_per_cluster.keys()) != set(other.points_per_cluster.keys()):
            print("Points per cluster keys are not equal.")
            return False
        for cluster in self.points_per_cluster:
            if self.points_per_cluster[cluster] != other.points_per_cluster[cluster]:
                print(f"Points in cluster {cluster} are not equal.")
                return False
        # Check if selections per cluster are equal
        if set(self.selection_per_cluster.keys()) != set(other.selection_per_cluster.keys()):
            print("Selection per cluster keys are not equal.")
            return False
        for cluster in self.selection_per_cluster:
            if self.selection_per_cluster[cluster] != other.selection_per_cluster[cluster]:
                print(f"Selection in cluster {cluster} is not equal.")
                return False
        # Check if non-selections per cluster are equal
        if set(self.nonselection_per_cluster.keys()) != set(other.nonselection_per_cluster.keys()):
            print("Non-selection per cluster keys are not equal.")
            return False
        for cluster in self.nonselection_per_cluster:
            if self.nonselection_per_cluster[cluster] != other.nonselection_per_cluster[cluster]:
                print(f"Non-selection in cluster {cluster} is not equal.")
                return False
        # Check if closest intra cluster distances are equal
        try:
            if not np.allclose(self.closest_distances_intra, other.closest_distances_intra, atol=PRECISION_THRESHOLD):
                print("Closest intra cluster distances are not equal.")
                return False
        except:
            print("Closest intra cluster distances could not be compared.")
            return False
        # Check if closest intra cluster points are equal
        try:
            if not np.array_equal(self.closest_points_intra, other.closest_points_intra):
                print("Closest intra cluster points are not equal.")
                return False
        except:
            print("Closest intra cluster points could not be compared.")
            return False
        # Check if closest inter cluster distances are equal
        try:
            if not np.allclose(self.closest_distances_inter, other.closest_distances_inter, atol=PRECISION_THRESHOLD):
                print("Closest inter cluster distances are not equal.")
                return False
        except:
            print("Closest inter cluster distances could not be compared.")
            return False
        # Check if closest inter cluster points are equal
        try:
            if not np.array_equal(self.closest_points_inter, other.closest_points_inter):
                print("Closest inter cluster points are not equal.")
                print(self.closest_points_inter)
                print(other.closest_points_inter)
                return False
        except:
            print("Closest inter cluster points could not be compared.")
            return False
        # Check if feasibilities are equal
        if self.feasible != other.feasible:
            print("Feasibilities are not equal.")
            return False
        # Check if objectives are equal
        if not math.isclose(self.objective, other.objective, rel_tol=PRECISION_THRESHOLD):
            print("Objectives are not equal.")
            return False

        return True

    # Objective decomposition
    def decompose_objective(self, selection_cost: float):
        """
        Calculates the objective value of the solution decomposed into:
            - selection cost
            - intra cluster distance costs
            - inter cluster distance costs
        In addition, this method allows for another selection cost to be applied which
        prevents having to re-initialize a Solution object
        NOTE: If selection is not feasible, the objective value is set to np.inf
        and some of the internal attributes will not be set.

        Parameters:
        -----------
        selection_cost: float
            The cost associated with selecting a point.
            NOTE: for now this does not allow for custom definitions relating the selection
            cost to the number of points in a cluster for example.

        Returns:
        --------
        dict
            A dictionary with the following keys:
            - "selection": the total selection cost
            - "intra": the total intra cluster distance cost
            - "inter": the total inter cluster distance cost
            If the solution is not feasible, returns None.
        """
        is_feasible = self.determine_feasibility()
        cost = {
            "selection": 0.0,
            "intra": 0.0,
            "inter": 0.0
        }
        if not is_feasible:
            return None
        # Selection costs
        for idx in np.where(self.selection)[0]:
            cost["selection"] += selection_cost
        # Intra cluster distance costs
        for cluster in self.unique_clusters:
            for idx in self.nonselection_per_cluster[cluster]:
                cur_min = AUXILIARY_DISTANCE_DTYPE(np.inf)
                for other_idx in sorted(list(self.selection_per_cluster[cluster])): #this is to ensure consistent ordering
                    cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                    if cur_dist < cur_min:
                        cur_min = cur_dist
                cost["intra"] += cur_min
        # Inter cluster distance costs
        for cluster_1, cluster_2 in itertools.combinations(self.unique_clusters, 2):
            cur_max = -AUXILIARY_DISTANCE_DTYPE(np.inf)
            for point_1 in sorted(list(self.selection_per_cluster[cluster_1])): #this is to ensure consistent ordering
                for point_2 in sorted(list(self.selection_per_cluster[cluster_2])): #this is to ensure consistent ordering
                    cur_dist = 1.0 - get_distance(point_1, point_2, self.distances, self.num_points)
                    if cur_dist > cur_max:
                        cur_max = cur_dist
            cost["inter"] += cur_max
        return cost

# Multiprocessing Solution class using shared memory (stable)
class Solution_shm(Solution):
    def __init__(self, distances, clusters: np.ndarray, selection: np.ndarray = None, selection_cost: float = 1.0, cost_per_cluster: int = 0, shm_prefix: str = None, seed=None):
        """
        Initialize a Solution object using shared memory arrays.
        
        Parameters:
        -----------
        distances_func: callable
            A function that yields (i, j, distance) tuples to populate the distance matrix.
            This allows streaming distances without creating a full copy in memory.
            The function should yield condensed distance matrix indices in order.
        clusters: numpy.ndarray
            A 1D array where clusters[i] represents the cluster assignment of point i.
        selection: numpy.ndarray, optional
            A 1D boolean array indicating which points are selected.
        selection_cost: float
            The cost associated with selecting a point.
        cost_per_cluster: int
            Defines how the cost of selecting a point in each cluster is calculated.
        shm_prefix: str, optional
            Prefix for shared memory segment names. If None, a unique prefix is generated.
        seed: int or np.random.RandomState, optional
            Random seed for reproducibility.
        """
        # Assert distances is a matrix or callable
        if not callable(distances) and not isinstance(distances, np.ndarray):
            raise TypeError("distances_func must be callable or a numpy array")
        # If distances is matrix, assert that distances and clusters have the same number of rows/points
        if isinstance(distances, np.ndarray):
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

        # Initialize basic attributes
        self.num_points = clusters.shape[0]
        self.selection_cost = selection_cost

        ################################ SHARED MEMORY SETUP ################################
        # Generate unique prefix for shared memory
        if shm_prefix is None:
            import uuid
            shm_prefix = f"sol_{uuid.uuid4().hex[:8]}_"
        self.shm_prefix = shm_prefix
        
        # Store shared memory handles for cleanup
        self._shm_handles = {}
        
        # Create shared memory for clusters and populate
        unique_clusters, inv = np.unique(clusters, return_inverse=True)
        self._create_shm_array("clusters", clusters.shape, np.int64)
        self.clusters[:] = inv.astype(np.int64)
        self._create_shm_array("unique_clusters", unique_clusters.shape, np.int64)
        self.unique_clusters[:] = np.arange(unique_clusters.shape[0], dtype=np.int64)
        self.original_clusters = unique_clusters #store original cluster ids for reference
        self._create_shm_array("num_selected_per_cluster", (unique_clusters.shape[0],), np.int64) #used to track feasibility of removal moves
        self.num_clusters = unique_clusters.shape[0]
        
        # Calculate condensed distance matrix size
        condensed_size = (self.num_points * (self.num_points - 1)) // 2
        
        # Create shared memory for distances and stream data directly
        self._create_shm_array("distances", (condensed_size,), DISTANCE_DTYPE)
        
        # If distances is an array, copy directly
        if isinstance(distances, np.ndarray):
            # Copy distances into shared memory
            flat_distances = squareform(distances, force="tovector", checks=False)
            np.copyto(self.distances, flat_distances.astype(dtype=DISTANCE_DTYPE))
        else: #otherwise stream distances into shared memory
            for i, j, dist in distances():
                if not (0 <= dist <= 1):
                    raise ValueError(f"Distance at ({i}, {j}) = {dist} is not in range [0, 1].")
                idx = get_index(i, j, self.num_points)
                self.distances[idx] = dist
        
        # Create shared memory for selection
        self._create_shm_array("selection", selection.shape, bool)
        np.copyto(self.selection, selection.astype(dtype=bool))

        # Create shared memory for auxiliary arrays
        self._create_shm_array("cost_per_cluster", (self.unique_clusters.shape[0],), AUXILIARY_DISTANCE_DTYPE)
        self._create_shm_array('closest_distances_intra', (self.num_points,), AUXILIARY_DISTANCE_DTYPE)
        self._create_shm_array('closest_points_intra', (self.num_points,), np.int32)
        self._create_shm_array('closest_distances_inter', (self.num_clusters, self.num_clusters), AUXILIARY_DISTANCE_DTYPE)
        self._create_shm_array('closest_points_inter', (self.num_clusters, self.num_clusters), np.int32)

        ################################ Initialize cost per cluster ################################
        # Calculate cost per cluster
        for cluster in self.unique_clusters:
            if cost_per_cluster == 0: #default behavior, set to selection cost
                self.cost_per_cluster[cluster] = selection_cost
            elif cost_per_cluster == 1: #set to 1 / number of points in cluster
                self.cost_per_cluster[cluster] = selection_cost / np.sum(self.clusters == cluster)
            elif cost_per_cluster == 2:
                # Define the average distance in a cluster as the average distance
                # of all points in the cluster to the centroid of the cluster.
                cluster_points = np.where(self.clusters == cluster)[0]
                centroid_idx = np.argmin([np.sum([get_distance(p, q, self.distances, self.num_points) for q in cluster_points]) for p in cluster_points])
                centroid = cluster_points[centroid_idx]
                self.cost_per_cluster[cluster] = np.mean([get_distance(centroid, p, self.distances, self.num_points) for p in cluster_points])
            elif cost_per_cluster == -2:
                # Define the average distance in a cluster as the average similarity
                # of all points in the cluster to the centroid of the cluster.
                cluster_points = np.where(self.clusters == cluster)[0]
                centroid_idx = np.argmin([np.sum([get_distance(p, q, self.distances, self.num_points) for q in cluster_points]) for p in cluster_points])
                centroid = cluster_points[centroid_idx]
                self.cost_per_cluster[cluster] = selection_cost * (1.0 - np.mean([get_distance(centroid, p, self.distances, self.num_points) for p in cluster_points]))
                # Define the average distance in a cluster as the average distance
                # of all points in the cluster to the closest pointn in the cluster.
            elif cost_per_cluster == 3:
                cluster_points = np.where(self.clusters == cluster)[0]
                self.cost_per_cluster[cluster] = np.mean([np.min([get_distance(point, p, self.distances, self.num_points) for p in cluster_points if p != point]) for point in cluster_points])      
        
        # Build CSR representation for clusters
        self._create_shm_array("cluster_members", (self.num_points,), np.int64)
        self._create_shm_array("cluster_offsets", (self.unique_clusters.shape[0]+1,), np.int64)

        order = np.argsort(self.clusters)
        np.copyto( self.cluster_members, order.astype(np.int64, copy=False))

        counts = np.bincount(self.clusters, minlength=self.unique_clusters.shape[0]).astype(np.int64)
        self.cluster_offsets[1:] = np.cumsum(counts)

        # Calculate objective
        self._create_shm_array("objective", (1,), np.float64)
        self.calculate_objective()

        # Create epoch counter
        self._create_shm_array("epoch", (1,), np.int64)
    
    @classmethod
    def attach(cls, shm_prefix: str, num_points: int, num_clusters: int):
        """
        Creates a Solution_shm instance by attaching to existing shared memory blocks.
        NOTE: This method assumes that the shared memory segments have already been created
        and populated by another process.

        Parameters:
        -----------
        shm_prefix: str
            The prefix used for the shared memory segments.
        num_points: int
            The number of points in the solution.
        num_clusters: int
            The number of clusters in the solution.

        Returns:
        Solution_shm
            An instance of Solution_shm attached to the existing shared memory.
        """
        self = cls.__new__(cls)  # Create an uninitialized instance
        self.shm_prefix = shm_prefix
        self.num_points = num_points
        self.num_clusters = num_clusters
        self._shm_handles = {}

        def _attach(name: str, shape: tuple, dtype):
            """
            Helper function for attaching to a shared memory array.
            """
            shm_name = f"{self.shm_prefix}{name}"
            shm_handle = shm.SharedMemory(create=False, name=shm_name)
            self._shm_handles[name] = shm_handle
            arr = np.ndarray(shape, dtype=dtype, buffer=shm_handle.buf)
            setattr(self, name, arr)

        condensed_size = (num_points * (num_points - 1)) // 2

        # Attach cluster-related arrays
        _attach("clusters", (num_points,), np.int64)
        _attach("unique_clusters", (num_clusters,), np.int64)
        _attach("num_selected_per_cluster", (num_clusters,), np.int64)

        # Attach distances and selection
        _attach("distances", (condensed_size,), DISTANCE_DTYPE)
        _attach("selection", (num_points,), bool)

        # Attach auxiliary arrays
        _attach("cost_per_cluster", (num_clusters,), AUXILIARY_DISTANCE_DTYPE)
        _attach('closest_distances_intra', (num_points,), AUXILIARY_DISTANCE_DTYPE)
        _attach('closest_points_intra', (num_points,), np.int32)
        _attach('closest_distances_inter', (num_clusters, num_clusters), AUXILIARY_DISTANCE_DTYPE)
        _attach('closest_points_inter', (num_clusters, num_clusters), np.int32)

        # Attach CSR representation
        _attach("cluster_members", (num_points,), np.int64)
        _attach("cluster_offsets", (num_clusters + 1,), np.int64)

        # Attach epoch counter
        _attach("epoch", (1,), np.int64)

        # Attach objective
        _attach("objective", (1,), np.float64)

        self.num_points = num_points
        self.num_clusters = num_clusters
        self.feasible = True  # Default to True

        return self
        
    # shm helpers
    def _create_shm_array(self, name: str, shape: tuple, dtype):
        """
        Creates a shared memory array and stores the handle.
        
        Parameters:
        -----------
        name: str
            Name suffix for the shared memory segment.
        shape: tuple
            Shape of the array.
        dtype: numpy dtype
            Data type of the array.
        """
        shm_name = f"{self.shm_prefix}{name}"
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        
        # Create shared memory
        shm_handle = shm.SharedMemory(create=True, size=size, name=shm_name)
        self._shm_handles[name] = shm_handle
        
        # Create numpy array backed by shared memory
        arr = np.ndarray(shape, dtype=dtype, buffer=shm_handle.buf)
        arr[:] = 0 #initialize to 0
        setattr(self, name, arr)
    
    def __enter__(self):
        """
        Context manager so shared memory can be cleaned up automatically!
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Cleanup shared memory on exiting context.
        """
        self.cleanup()
        return False #do not suppress exceptions

    def cleanup(self):
        """
        Cleanup shared memory resources.
        Should be called when the solution object is no longer needed.
        """
        for name, shm_handle in self._shm_handles.items():
            try:
                shm_handle.close()
                shm_handle.unlink()
            except FileNotFoundError:
                pass  #already unlinked
            except Exception as e:
                print(f"Error cleaning up shared memory '{name}': {e}")
        self._shm_handles.clear()

    def close_only(self):
        """
        Close shared memory resources without unlinking.
        Useful when multiple processes are using the same shared memory.
        """
        for handle in self._shm_handles.values():
            try:
                handle.close()
            except FileNotFoundError:
                pass  #already closed
            except Exception as e:
                print(f"Error closing shared memory: {e}")
        self._shm_handles.clear()

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    # Core state and feasibility methods
    def calculate_objective(self):
        """
        Calculates the objective value of the solution, as well as set all the
        inter and intra cluster distances and points.
        NOTE: If selection is not feasible, the objective value is set to np.inf
        and some of the internal attributes will not be set.
        """
        # Initialize closest distances and points arrays
        #INTRA-CLUSTER
        self.closest_distances_intra.fill(0)
        self.closest_points_intra[:] = np.arange(self.num_points, dtype=np.int32)
        #INTER-CLUSTER
        self.closest_distances_inter.fill(0)
        self.closest_points_inter.fill(0)
        """
        Interpretation of closest_points_inter: given a pair of clusters (cluster1, cluster2),
        the value at closest_points_inter[cluster1, cluster2] is the index of the point in cluster1 that is closest to any point in cluster2.
        In principle this thus assumes that the leading index is the "from" cluster and thus yields
        the point in that cluster that is closest to any point in cluster2 (which can be retrieved from closest_points_inter[cluster2, cluster1]).
        """
        # Check feasibility
        is_feasible = self.determine_feasibility()
        if not is_feasible:
            self.feasible = False
            self.objective[0] = np.float64(np.inf)
            return self.objective
        self.feasible = True

        # Calculate objective value
        objective = 0.0

        self.num_selected_per_cluster.fill(0)
        # Selection cost
        for idx in np.flatnonzero(self.selection):
            objective += self.cost_per_cluster[self.clusters[idx]]
            self.num_selected_per_cluster[self.clusters[idx]] += 1

        # Intra-cluster distance costs
        for cluster in self.unique_clusters:
            for idx in self.iter_unselected(cluster):
                cur_min = AUXILIARY_DISTANCE_DTYPE(np.inf)
                cur_idx = None #index of the closest selected point of the same cluster
                for other_idx in self.iter_selected(cluster):
                    cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                    if cur_dist < cur_min:
                        cur_min = cur_dist
                        cur_idx = other_idx
                self.closest_distances_intra[idx] = AUXILIARY_DISTANCE_DTYPE(cur_min)
                self.closest_points_intra[idx] = np.int32(cur_idx)
                objective += cur_min

        #Inter-cluster distance costs
        for cluster_1, cluster_2 in itertools.combinations(self.unique_clusters, 2):
            cur_max = -AUXILIARY_DISTANCE_DTYPE(np.inf)
            cur_pair = (None, None) #indices of the closest selected points of the two clusters
            for point_1 in self.iter_selected(cluster_1):
                for point_2 in self.iter_selected(cluster_2):
                    cur_dist = 1.0 - get_distance(point_1, point_2, self.distances, self.num_points) #convert to similarity
                    if cur_dist > cur_max:
                        cur_max = AUXILIARY_DISTANCE_DTYPE(cur_dist)
                        cur_pair = (point_1, point_2)
            self.closest_distances_inter[cluster_1, cluster_2] = cur_max
            self.closest_distances_inter[cluster_2, cluster_1] = cur_max
            self.closest_points_inter[cluster_1, cluster_2] = np.int32(cur_pair[0])
            self.closest_points_inter[cluster_2, cluster_1] = np.int32(cur_pair[1])
            objective += cur_max

        self.objective[0] = np.float64(objective)
        return self.objective

    # Cluster iterator methods (needed due to CSR representation)
    def iter_cluster_members(self, cluster: int):
        """
        Generator that yields the indices of points in a given cluster.
        
        Parameters:
        -----------
        cluster: int
            The cluster for which to yield member indices.
        
        Yields:
        -------
        idx: int
            Indices of points in the specified cluster.
        """
        cluster_idx = np.where(self.unique_clusters == cluster)[0][0]
        start = self.cluster_offsets[cluster_idx]
        end = self.cluster_offsets[cluster_idx + 1]
        for idx in self.cluster_members[start:end]:
            yield idx

    def iter_selected(self, cluster: int):
        """
        Generator that yields the indices of selected points in a given cluster.
        
        Parameters:
        -----------
        cluster: int
            The cluster for which to yield selected member indices.
        
        Yields:
        -------
        idx: int
            Indices of selected points in the specified cluster.
        """
        for idx in self.iter_cluster_members(cluster):
            if self.selection[idx]:
                yield idx

    def iter_unselected(self, cluster: int):
        """
        Generator that yields the indices of unselected points in a given cluster.

        Parameters:
        -----------
        cluster: int
            The cluster for which to yield unselected member indices.

        Yields:
        -------
        idx: int
            Indices of unselected points in the specified cluster.
        """
        for idx in self.iter_cluster_members(cluster):
            if not self.selection[idx]:
                yield idx

    # Local search evaluation and acceptance methods
    def evaluate_add(self, idx_to_add: int, local_search: bool = False, stop_event = None):
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
        stop_event: multiprocessing.Event, optional
            An optional event that can be used to signal early termination of the evaluation.
            If the event is set during evaluation, the method will return (np.inf, None, None).
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the addition.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
            NOTE: new_closest_point will always be idx_to_add.
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, (point_in_this_cluster, point_in_other_cluster), new_distance)]
            NOTE: point_in_this_cluster will always be idx_to_add.
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        if self.selection[idx_to_add]:
            raise ValueError("The point to add must not be selected.")
        
        # Check early stop event
        if stop_event is not None and stop_event.is_set():
            return np.inf, None, None

        # Find current cluster
        cluster = self.clusters[idx_to_add]

        # Calculate selection cost
        original_objective = self.objective[0]
        candidate_objective = self.objective[0] + self.cost_per_cluster[cluster] # cost for adding the point

        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for i, idx in enumerate(self.iter_unselected(cluster)):
            if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                return np.inf, None, None
            cur_dist = get_distance(idx, idx_to_add, self.distances, self.num_points) # distance to current point (idx)
            if cur_dist < self.closest_distances_intra[idx]:
                candidate_objective += cur_dist - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, idx_to_add, cur_dist))

        # NOTE: Inter-cluster distances can only increase when adding a point, so when doing local search we can exit here if objective is worse
        if candidate_objective > original_objective and np.abs(candidate_objective - original_objective) > PRECISION_THRESHOLD and local_search:
            return np.inf, None, None

        # Calculate inter-cluster distances for other clusters
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        for other_cluster in self.unique_clusters:
            if stop_event is not None and stop_event.is_set():
                return np.inf, None, None
            if other_cluster != cluster:
                cur_max = self.closest_distances_inter[cluster, other_cluster]
                cur_idx = -1
                for i, idx in enumerate(self.iter_selected(other_cluster)):
                    if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                        return np.inf, None, None
                    cur_similarity = 1 - get_distance(idx, idx_to_add, self.distances, self.num_points) #this is the similarity, if it is more similar then change solution
                    if cur_similarity > cur_max:
                        cur_max = cur_similarity
                        cur_idx = idx
                if cur_idx > -1:
                    candidate_objective += cur_max - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, (idx_to_add, cur_idx), cur_max))

        return candidate_objective, add_within_cluster, add_for_other_clusters

    def evaluate_swap(self, idxs_to_add, idx_to_remove: int, stop_event = None):
        """
        Evaluates the effect of swapping a selected point for a/multiple unselected point(s)
        in the solution.

        Parameters:
        -----------
        idxs_to_add: tuple of int or list of int
            The index or indices of the point(s) to be added.
        idx_to_remove: int
            The index of the point to be removed.
        stop_event: multiprocessing.Event, optional
            An optional event that can be used to signal early termination of the evaluation.
            If the event is set during evaluation, the method will return (np.inf, None, None).
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the addition.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, (point_in_this_cluster, point_in_other_cluster), new_distance)]
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate addition.")
        try:
            num_to_add = len(idxs_to_add)
        except TypeError: #assumption is that this is an int
            num_to_add = 1
            idxs_to_add = [idxs_to_add]
        for idx in idxs_to_add:
            if self.selection[idx]:
                raise ValueError("The points to add must not be selected.")
        if not self.selection[idx_to_remove]:
            raise ValueError("The point to remove must be selected.")
        cluster = self.clusters[idx_to_remove]
        for idx in idxs_to_add:
            if self.clusters[idx] != cluster:
                raise ValueError("All points must be in the same cluster.")
            
        # Check early stop event
        if stop_event is not None and stop_event.is_set():
            return np.inf, None, None
            
        # Generate pool of alternative points to compare to
        new_selection = set(self.iter_selected(cluster))
        for idx in idxs_to_add:
            new_selection.add(idx)
        new_selection.remove(idx_to_remove)
        new_nonselection = set(self.iter_unselected(cluster))
        new_nonselection.add(idx_to_remove)

        # Calculate selection cost
        candidate_objective = self.objective[0] + (num_to_add - 1) * self.cost_per_cluster[cluster] #cost for swapping points

        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for i, idx in enumerate(new_nonselection):
            if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                return np.inf, None, None
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
            else: #point to be removed is not closest, check if one of newly added points is closer
                cur_dists = [(get_distance(idx, idx_to_add, self.distances, self.num_points), idx_to_add) for idx_to_add in idxs_to_add]
                cur_dist, idx_to_add = min(cur_dists, key=lambda x: x[0])
                if cur_dist < cur_closest_distance:
                    candidate_objective += cur_dist - cur_closest_distance
                    add_within_cluster.append((idx, idx_to_add, cur_dist))

        # Calculate inter-cluster distances for all other clusters
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        for other_cluster in self.unique_clusters:
            if stop_event is not None and stop_event.is_set():
                return np.inf, None, None
            if other_cluster != cluster:
                cur_closest_similarity = self.closest_distances_inter[cluster, other_cluster]
                cur_closest_point = self.closest_points_inter[cluster, other_cluster]
                cur_closest_pair = (-1, -1)
                if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                    cur_closest_similarity = -np.inf
                    for i, idx in enumerate(self.iter_selected(other_cluster)):
                        if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                            return np.inf, None, None
                        for other_idx in new_selection:
                            cur_similarity = 1.0 - get_distance(idx, other_idx, self.distances, self.num_points)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                cur_closest_pair = (other_idx, idx)
                else: #point to be removed is not closest, check if one of newly added points is closer
                    for i, idx in enumerate(self.iter_selected(other_cluster)):
                        if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                            return np.inf, None, None
                        cur_similarities = [(1.0 - get_distance(idx, idx_to_add, self.distances, self.num_points), idx_to_add) for idx_to_add in idxs_to_add]
                        cur_similarity, idx_to_add = max(cur_similarities, key = lambda x: x[0])
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            cur_closest_pair = (idx_to_add, idx)
                if cur_closest_pair[0] > -1:
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))

        return candidate_objective, add_within_cluster, add_for_other_clusters

    def evaluate_remove(self, idx_to_remove: int, local_search: bool = False, stop_event = None):
        """
        Evaluates the effect of removing a selected point from the solution.

        Parameters:
        -----------
        idx_to_remove: int
            The index of the point to be removed.
        local_search: bool
            If True, the method will return (np.inf, None, None) if the candidate objective
            is worse than the current objective, allowing for local search to skip unnecessary evaluations.
            If False, it will always evaluate the removal.
        stop_event: multiprocessing.Event, optional
            An optional event that can be used to signal early termination of the evaluation.
            If the event is set during evaluation, the method will return (np.inf, None, None).
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the removal.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, (point_in_this_cluster, point_in_other_cluster), new_distance)]
        """
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot evaluate removal.")
        if not self.selection[idx_to_remove]:
            raise ValueError("The point to remove must be selected.")

        # Check early stop event
        if stop_event is not None and stop_event.is_set():
            return np.inf, None, None
        
        # Find current cluster
        cluster = self.clusters[idx_to_remove]

        # Generate pool of alternative points to compare to
        new_selection = set(self.iter_selected(cluster))
        new_selection.remove(idx_to_remove)
        new_nonselection = set(self.iter_unselected(cluster))
        new_nonselection.add(idx_to_remove)

        # Calculate selection cost
        original_objective = self.objective[0]
        candidate_objective = self.objective[0] - self.cost_per_cluster[cluster]

        # Calculate inter-cluster distances for all other clusters
        # NOTE: Intra-cluster distances can only increase when removing a point, Thus if inter-cluster distances
        # increase, we can exit early.
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                cur_closest_similarity = self.closest_distances_inter[cluster, other_cluster]
                cur_closest_point = self.closest_points_inter[cluster, other_cluster]
                cur_closest_pair = (-1, -1)
                if cur_closest_point == idx_to_remove:
                    cur_closest_similarity = -np.inf
                    for i, idx in enumerate(self.iter_selected(other_cluster)):
                        if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                            return np.inf, None, None
                        for other_idx in new_selection:
                            cur_similarity = 1.0 - get_distance(idx, other_idx, self.distances, self.num_points)
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
        
        # NOTE: Intra-cluster distances can only increase when removing a point, so when doing local search we can exit here if objective is worse
        if candidate_objective > original_objective and np.abs(candidate_objective - original_objective) > PRECISION_THRESHOLD and local_search:
            return np.inf, None, None
        
        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for i, idx in enumerate(new_nonselection):
            if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                return np.inf, None, None
            cur_closest_point = self.closest_points_intra[idx]
            if cur_closest_point == idx_to_remove:
                cur_closest_distance = np.inf
                for j, other_idx in enumerate(new_selection):
                    if stop_event is not None and (j & 63)==0 and stop_event.is_set(): #check every 64 iterations
                        return np.inf, None, None
                    if other_idx != idx:
                        cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                        if cur_dist < cur_closest_distance:
                            cur_closest_distance = cur_dist
                            cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
        
        return candidate_objective, add_within_cluster, add_for_other_clusters

    def accept_move(self, idxs_to_add: list, idxs_to_remove: list, candidate_objective: float, add_within_cluster: list, add_for_other_clusters: list):
        """
        Accepts a move to the solution, where multiple points can be added and removed at once.
        NOTE: This assumes that the initial solution and the move
        are feasible and will not check for this.

        PARAMETERS:
        -----------
        idxs_to_add: list of int
            The indices of the points to be added.
            NOTE: This assumes that all indices to be added are in the same cluster (which should be the same as the indices to remove)!
        idxs_to_remove: list of int
            The indices of the points to be removed.
            NOTE: This assumes that all indices to be removed are in the same cluster (which should be the same as the indices to add)!
        candidate_objective: float
            The objective value of the solution after the move.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, (closest_point_this_cluster, closest_point_other_cluster), new_distance)]
        """
        found_clusters = set()
        for idx in idxs_to_add + idxs_to_remove:
            found_clusters.add(self.clusters[idx])
        if len(found_clusters) != 1:
            raise ValueError("All points to add and remove must be in the same cluster.")
        cluster = found_clusters.pop()
        # Updating state attributes of this solution object
        for idx_to_add in idxs_to_add:
            self.selection[idx_to_add] = True
            self.num_selected_per_cluster[cluster] += 1
        for idx_to_remove in idxs_to_remove:
            self.selection[idx_to_remove] = False
            self.num_selected_per_cluster[cluster] -= 1
        # Updating intra-cluster distances and points
        for idx_to_change, new_closest_point, new_distance in add_within_cluster:
            self.closest_distances_intra[idx_to_change] = new_distance
            self.closest_points_intra[idx_to_change] = new_closest_point
        # Updating inter-cluster distances and points
        for other_cluster, (closest_point_this_cluster, closest_point_other_cluster), new_distance in add_for_other_clusters:
            self.closest_distances_inter[cluster, other_cluster] = new_distance
            self.closest_distances_inter[other_cluster, cluster] = new_distance
            self.closest_points_inter[cluster, other_cluster] = closest_point_this_cluster
            self.closest_points_inter[other_cluster, cluster] = closest_point_other_cluster

        self.objective[0] = candidate_objective

    # Local search move generation methods
    def generate_indices_add(self, random: bool = False):
        """
        Generates indices of points that can be added to the solution.

        Parameters:
        -----------
        random: bool
            If True, the order of indices is randomized. Default is False.
            NOTE: this uses the random state stored in the Solution object.

        Yields:
        -------
        idx: int
            Indices of points that can be added to the solution.
        """
        indices = np.flatnonzero(~self.selection)
        if random:
            yield from self.random_state.permutation(indices)
        else:
            yield from indices

    def generate_indices_swap_old(self, number_to_add: int = 1, random: bool = False):
        """
        Generates indices of pairs of points that can be swapped in the solution.
        NOTE: when running in random mode, we randomly iterate over clusters, and
        indices in a cluster.
        NOTE: THIS VERSION IS DEPRECATED, USE generate_indices_swap INSTEAD!
        
        Parameters:
        -----------
        number_to_add: int
            The number of points to add in the swap operation. Default is 1 (remove 1 point, add 1 point).
        random: bool
            If True, the order of indices is randomized. Default is False.
            NOTE: this uses the random state stored in the Solution object.
            NOTE: although the cluster order can be randomized, the cluster is exhausted before moving to the next cluster.

        Yields:
        -------
        idxs_to_add: list of int
            Indices of points to add to the solution.
        idx_to_remove: int
            Index of point to remove from the solution.
        """
        if random:
            cluster_order = self.random_state.permutation(self.unique_clusters)
        else:
            cluster_order = self.unique_clusters
        for cluster in cluster_order:
            clusters_mask = self.clusters == cluster
            selected = np.flatnonzero(clusters_mask & self.selection)
            unselected = np.flatnonzero(clusters_mask & ~self.selection)

            if random:
                if selected.size == 0 or unselected.size == 0: #skip permuting if no points to swap
                    continue
                selected = self.random_state.permutation(selected)
                unselected = self.random_state.permutation(unselected)

            for idx_to_remove in selected:
                if number_to_add == 1:
                    for idx_to_add in unselected:
                        yield [idx_to_add], idx_to_remove
                else:
                    for indices_to_add in itertools.combinations(unselected, number_to_add):
                        yield list(indices_to_add), idx_to_remove

    def generate_indices_swap(self, number_to_add: int = 1, random: bool = False):
        """
        Creates a generator for every cluster, so that
        clusters can be exhausted in random order (opposed to exhausting one cluster at a time).

        Parameters:
        -----------
        number_to_add: int
            The number of points to add in the swap operation. Default is 1 (remove 1 point, add 1 point).
        random: bool
            If True, the order of indices is randomized. Default is False.
            NOTE: this uses the random state stored in the Solution object.
            NOTE: although the cluster order can be randomized, it exhausts all swaps for a given remove index
            in randomized order.

        Yields:
        -------
        idxs_to_add: list of int
            Indices of points to add to the solution.
        idx_to_remove: int
            Index of point to remove from the solution.
        """            
        cluster_iterators = {}
        for cluster in self.unique_clusters:
            clusters_mask = self.clusters == cluster
            selected = np.flatnonzero(clusters_mask & self.selection)
            unselected = np.flatnonzero(clusters_mask & ~self.selection)
            cluster_iterators[cluster] = self._generate_swaps_in_cluster(selected, unselected, number_to_add, random)

        remaining_clusters = list(cluster_iterators.keys())
        while remaining_clusters:
            # With random, randomly select a cluster to yield from
            if random:
                try:
                    cur_cluster = self.random_state.choice(remaining_clusters)
                    yield next( cluster_iterators[cur_cluster] )
                except StopIteration:
                    cluster_iterators.pop(cur_cluster)
                    remaining_clusters.remove(cur_cluster)
            # In non-random, just go through clusters in order
            else:
                cur_cluster = remaining_clusters[0]
                while True:
                    try:
                        yield next( cluster_iterators[cur_cluster] )
                    except StopIteration:
                        cluster_iterators.pop(cur_cluster)
                        remaining_clusters.remove(cur_cluster)
                        break

    def _generate_swaps_in_cluster(self, selected: np.ndarray, unselected: np.ndarray, number_to_add: int, random: bool = False):
        """
        Helper function to generate swaps within a cluster.

        Parameters:
        -----------
        selected: numpy.ndarray
            Indices of selected points in the cluster.
        unselected: numpy.ndarray
            Indices of unselected points in the cluster.
        number_to_add: int
            The number of points to add in the swap operation.
        random: bool
            If True, the order of indices is randomized. Default is False.

        Yields:
        -------
        idxs_to_add: list of int
            Indices of points to add to the solution.
        idx_to_remove: int
            Index of point to remove from the solution.
        """
        if selected.size == 0 or unselected.size < number_to_add: #skip permuting if no points to swap
            return #empty generator

        if random:
            selected = self.random_state.permutation(selected)
            unselected = self.random_state.permutation(unselected)

        for idx_to_remove in selected:
            if number_to_add == 1:
                for idx_to_add in unselected:
                    yield [idx_to_add], idx_to_remove
            else:
                for indices_to_add in itertools.combinations(unselected, number_to_add):
                    yield list(indices_to_add), idx_to_remove

    def generate_indices_remove(self, random: bool = False):
        """
        Generates indices of points that can be removed from the solution.

        Parameters:
        -----------
        random: bool
            If True, the order of indices is randomized. Default is False.
            NOTE: This uses the random state stored in the Solution object.

        Yields:
        -------
        idx: int
            Indices of points that can be removed from the solution.
        """
        indices = np.flatnonzero(self.selection)
        if random:
            for idx in self.random_state.permutation(indices):
                if self.num_selected_per_cluster[self.clusters[idx]] > 1:
                    yield idx
        else:
            for idx in indices:
                if self.num_selected_per_cluster[self.clusters[idx]] > 1:
                    yield idx

    # Local search (multiprocessing, for single processing see Solution)
    def local_search(self,
                    num_processes: int = 2,
                    max_iterations: int = 10_000, max_runtime: float = np.inf,
                    random_move_order: bool = True, random_index_order: bool = True, move_order: list = ["add", "swap", "doubleswap", "remove"],
                    doubleswap_time_threshold: float = 60.0,
                    task_queue_size: int = 2000,
                    logging: bool = False, logging_frequency: int = 500,
                    ):
        """
        Perform local search to find a (local) optimal solution using multiple processes.
        The core implementation here is that the main process generates candidate moves and
        distributes them to worker processes for evaluation. When a worker finds an improving move,
        it sends it back to the main process which then updates the solution and notifies all workers
        to restart their search from the new solution.

        Parameters:
        -----------
        num_processes: int
            The number of worker processes to use for local search. Default is 2.
        max_iterations: int
            The maximum number of iterations to perform. Default is 10,000.
        max_runtime: float
            The maximum runtime in seconds for the local search. Default is infinity.
        random_move_order: bool
            If True, the order of moves (add, swap, doubleswap,
            remove) is randomized. Default is True.
        random_index_order: bool
            If True, the order of indices for each move type is randomized. Default is True.
            NOTE: if random_move_order is True, but this is false,
            all moves of a particular type will be exhausted before moving to the next type,
            but the order of moves is random.
        move_order: list of str
            If provided, this list will be used to determine the order of moves. If random_move_order
            is True, this list will be shuffled before use.
            NOTE: this list should only contain the following move types (as strings):
                - "add"
                - "swap"
                - "doubleswap"
                - "remove"
            NOTE: by leaving out a move type, it will not be considered in the local search.
        doubleswap_time_threshold: float
            The time threshold in seconds after which double swap moves will no longer be considered.
            Default is 60.0 seconds.
            NOTE: this is on a per-iteration basis, so if an iteration takes longer than this threshold,
            doubleswaps will be skipped in current iteration, but re-added for the next iteration.
        task_queue_size: int
            The maximum size of the task queue used to distribute evaluation tasks to worker processes.
            Default is 2000.
        logging: bool
            If True, information about the local search will be printed. Default is False.
        logging_frequency: int
            If logging is True, information will be printed every logging_frequency iterations. Default is 500.

        Returns:
        --------
        time_per_ieration: list of floats
            The time taken for each iteration.
            NOTE: this is primarily for logging purposes
        objectives: list of floats
            The objective value after each iteration.
        """
        # Validate input
        if not isinstance(num_processes, int) or num_processes < 1:
            raise ValueError("num_processes must be an integer greater than 0.")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(random_move_order, bool):
            raise ValueError("random_move_order must be a boolean value.")
        if not isinstance(random_index_order, bool):
            raise ValueError("random_index_order must be a boolean value.")
        if not isinstance(move_order, list):
            raise ValueError("move_order must be a list of move types.")
        else:
            if len(move_order) == 0:
                raise ValueError("move_order must contain at least one move type.")
            valid_moves = {"add", "swap", "doubleswap", "remove"}
            if len(set(move_order) - valid_moves) > 0:
                raise ValueError("move_order must contain only the following move types: add, swap, doubleswap, remove.")
        if not isinstance(doubleswap_time_threshold, (int, float)) or doubleswap_time_threshold <= 0:
            raise ValueError("doubleswap_time_threshold must be a positive number.")
        if not isinstance(task_queue_size, int) or task_queue_size < 1:
            raise ValueError("task_queue_size must be a positive integer.")
        if not isinstance(logging, bool):
            raise ValueError("logging must be a boolean value.")
        if not isinstance(logging_frequency, int) or logging_frequency < 1:
            raise ValueError("logging_frequency must be a positive integer.")  
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")

        # Create epoch tag
        self.epoch[0] = 0

        # Create context variables and main process variables
        context = mp.get_context("spawn")
        stop_event = context.Event()
        task_q = context.Queue(maxsize=task_queue_size)
        result_q = context.Queue()
        ack_q = context.Queue()

        # Start worker processes (attach shared memory handles)
        workers = []
        for _ in range(num_processes):
            worker = context.Process(
                target = _shm_worker_main,
                args = (
                    self.shm_prefix,
                    self.num_points,
                    self.num_clusters,
                    task_q,
                    result_q,
                    ack_q,
                    stop_event,
                ),
                daemon = True,
            )
            worker.start()
            workers.append(worker)

        def drain_queue(q):
            """Helper function to drain a queue."""
            try:
                while True:
                    q.get_nowait()
            except Exception:
                return

        # Main local search loop
        try:
            iteration = 0
            time_per_iteration = []
            objectives = []
            solution_changed = False

            start_time = time.time()
            while iteration < max_iterations:
                if time.time() - start_time > max_runtime:
                    print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time:.2f} seconds). Stopping local search.", flush=True)
                    break
                current_iteration_time = time.time()
                objectives.append(self.objective[0])
                solution_changed = False
                stop_event.clear()
                drain_queue(result_q)


                # Create move generators for every movetype so doubleswaps can be removed if needed
                move_generator = {}
                for move_type in move_order:
                    if move_type == "add":
                        move_generator["add"] = self.generate_indices_add(random=random_index_order)
                    elif move_type == "swap":
                        move_generator["swap"] = self.generate_indices_swap(number_to_add=1, random=random_index_order)
                    elif move_type == "doubleswap":
                        move_generator["doubleswap"] = self.generate_indices_swap(number_to_add=2, random=random_index_order)
                    elif move_type == "remove":
                        move_generator["remove"] = self.generate_indices_remove(random=random_index_order)
                active_moves = move_order.copy() #list of move types for this iteration

                def next_task():
                    """
                    Helper function to get the next task from the move generators.
                    """
                    while active_moves:
                        if (time.time() - current_iteration_time) > doubleswap_time_threshold:
                            if "doubleswap" in active_moves:
                                print(f"Iteration {iteration}: Removed doubleswap moves due to time threshold exceeded ({time.time() - current_iteration_time:.2f} seconds).", flush=True)
                                active_moves.remove("doubleswap")
                                if not active_moves:
                                    return None, None

                        if random_move_order:
                            move_type = self.random_state.choice(active_moves)
                        else:
                            move_type = active_moves[0]
                        try:
                            move_content = next( move_generator[move_type] )
                        except StopIteration: #clear move from generator if no more moves are available
                            active_moves.remove(move_type)
                            del move_generator[move_type]
                            continue

                        if move_type == "add":
                            return MOVE_ADD, move_content
                        elif move_type == "swap":
                            return MOVE_SWAP, move_content
                        elif move_type == "doubleswap":
                            return MOVE_DSWAP, move_content
                        elif move_type == "remove":
                            return MOVE_REMOVE, move_content
                        
                    return None, None

                # Distribute tasks to workers and watch for first improving move
                while not solution_changed:
                    if time.time() - current_iteration_time > max_runtime:
                        break

                    # Check if workers have found improvements
                    try:
                        epoch, move_type, move_data, candidate_objective = result_q.get_nowait()
                        if epoch == self.epoch[0]: #only accept move if epoch matches
                            solution_changed = True
                            break #if move was submitted to result_q, break to process it
                    except Exception:
                        pass

                    # Distribute tasks to workers
                    move_code, move_content = next_task()
                    if move_code is None:
                        break #no more moves available
                    # Push task
                    task_q.put( (int(self.epoch[0]), move_code, move_content) )

                # If solution changed, process the move
                if solution_changed:
                    stop_event.set() #notify workers to stop current evaluations

                    # Drain task and result queue
                    drain_queue(task_q)
                    drain_queue(result_q)
                    drain_queue(ack_q)

                    # Acknowledge all workers have stopped
                    for _ in workers:
                        task_q.put( (int(self.epoch[0]), MOVE_SYNC, None) ) #send sync signal
                    for _ in workers:
                        while True:
                            a = ack_q.get() #wait for ack which puts current iteration
                            if a == int(self.epoch[0]):
                                break

                    # Accept the move by re-evaluating it to get the updates
                    if move_type == MOVE_ADD:
                        idx_to_add = move_data
                        new_objective, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add, local_search=False)
                        self.accept_move([idx_to_add], [], new_objective, add_within_cluster, add_for_other_clusters)
                    elif move_type == MOVE_SWAP:
                        idxs_to_add, idx_to_remove = move_data
                        new_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idxs_to_add, idx_to_remove)
                        self.accept_move(idxs_to_add, [idx_to_remove], new_objective, add_within_cluster, add_for_other_clusters)
                    elif move_type == MOVE_DSWAP:
                        idxs_to_add, idx_to_remove = move_data
                        new_objective, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idxs_to_add, idx_to_remove)
                        self.accept_move(idxs_to_add, [idx_to_remove], new_objective, add_within_cluster, add_for_other_clusters)
                    elif move_type == MOVE_REMOVE:
                        idx_to_remove = move_data
                        new_objective, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove, local_search=False)
                        self.accept_move([], [idx_to_remove], new_objective, add_within_cluster, add_for_other_clusters)

                    time_per_iteration.append(time.time() - current_iteration_time)
                    iteration += 1
                    self.epoch[0] += 1
                    if logging and (iteration % logging_frequency == 0):
                        print(f"Iteration {iteration}: Objective = {self.objective[0]:.10f}", flush=True)
                        print(f"Average runtime last {logging_frequency} iterations: {np.mean(time_per_iteration[-logging_frequency:]):.6f} seconds", flush=True)
                # Solution did not change -> exit local search
                else:
                    break

            return time_per_iteration, objectives
        finally:
            # Terminate workers
            stop_event.set()
            drain_queue(task_q)
            for _ in workers:
                try:
                    task_q.put( (int(self.epoch[0]), MOVE_STOP, None) )
                except Exception:
                    pass

            for worker in workers:
                worker.join(timeout=2.0)

    # Equality check
    def __eq__(self, other):
        """
        Check if two solutions are equal.
        NOTE: This purely checks if all attributes are equal, excluding the random state.
        NOTE: This is currently not finished!
        """
        if not isinstance(other, Solution_shm):
            return False
        return True #TODO: finish implementation

# Module-level functions
def _shm_worker_main(shm_prefix, num_points, num_clusters, task_q, result_q, ack_q, stop_event):
    """
    Worker function for multiprocessing evaluations.
    This function listens for tasks on the task queue, processes them,
    and puts the results on the result queue.

    Parameters:
    -----------
    shm_prefix: str
        The prefix for shared memory segments.
    num_points: int
        The number of points in the dataset.
    num_clusters: int
        The number of clusters in the dataset.
    task_q: multiprocessing.Queue
        The queue from which to receive tasks.
    result_q: multiprocessing.Queue
        The queue to which to send results.
    ack_q: multiprocessing.Queue
        The queue to which to send acknowledgments.
    stop_event: multiprocessing.Event
        An event that can be used to signal early termination.
    """
    global _WORKER_SOL
    _WORKER_SOL = Solution_shm.attach(shm_prefix, num_points, num_clusters)

    try:
        while True:
            epoch, move_code, move_args = task_q.get()

            # Check for termination signal
            if move_code == MOVE_STOP:
                break

            # Signal that worker is not inside evaluation
            if move_code == MOVE_SYNC:
                ack_q.put(epoch)
                continue

            # Terminate if stop event is set
            if stop_event.is_set():
                continue

            sol = _WORKER_SOL

            if sol.epoch is not None and epoch != sol.epoch[0]:
                # Solution has changed, skip current evaluation
                continue

            # Process move
            cur_obj = sol.objective[0]
            if move_code == MOVE_ADD:
                idx_to_add = move_args
                candidate_objective, _, _ = sol.evaluate_add(idx_to_add, local_search=True, stop_event=stop_event)
                if candidate_objective < cur_obj and abs(candidate_objective - cur_obj) > PRECISION_THRESHOLD:
                    # Suppress potentially late publishing of moves from old epochs
                    if (not stop_event.is_set()) and (sol.epoch is not None) and (epoch == sol.epoch[0]):
                        result_q.put( (epoch, move_code, idx_to_add, candidate_objective) )
            elif move_code == MOVE_SWAP or move_code == MOVE_DSWAP:
                idxs_to_add, idx_to_remove = move_args
                candidate_objective, _, _ = sol.evaluate_swap(idxs_to_add, idx_to_remove, stop_event=stop_event)
                if candidate_objective < cur_obj and abs(candidate_objective - cur_obj) > PRECISION_THRESHOLD:
                    # Suppress potentially late publishing of moves from old epochs
                    if (not stop_event.is_set()) and (sol.epoch is not None) and (epoch == sol.epoch[0]):
                        result_q.put( (epoch, move_code, (idxs_to_add, idx_to_remove), candidate_objective) )
            elif move_code == MOVE_REMOVE:
                idx_to_remove = move_args
                candidate_objective, _, _ = sol.evaluate_remove(idx_to_remove, local_search=True, stop_event=stop_event)
                if candidate_objective < cur_obj and abs(candidate_objective - cur_obj) > PRECISION_THRESHOLD:
                    # Suppress potentially late publishing of moves from old epochs
                    if (not stop_event.is_set()) and (sol.epoch is not None) and (epoch == sol.epoch[0]):
                        result_q.put( (epoch, move_code, idx_to_remove, candidate_objective) )
                
    finally:
        _WORKER_SOL.close_only()

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


############### EXPERIMENTAL MAIN FUNCTION ###############
import sourmash
from Bio import SeqIO
import time
from multiprocessing import Pool
import numpy as np
import math

def read_genomes():
    id2index = {}
    index2id = []
    lineages = []
    sequences = {}
    records = {}
    sequences_per_lineage = {}
    base_path = "/tudelft.net/staff-umbrella/refsetopt/SELECTION_PROJECT/SC2_example/GISAID_downloaded-24-05-2025_dates-01-09-2024_31-12-2024"
    for record in SeqIO.parse(f"{base_path}/sequences.fasta", "fasta"):
        cur_id = record.id
        cur_seq = str(record.seq)
        cur_idx = len(index2id)

        sequences[cur_id] = cur_seq
        records[cur_id] = record
        id2index[cur_id] = cur_idx
        index2id.append(cur_id)

    with open(f"{base_path}/metadata.tsv") as f:
        id_col = 0
        lineage_col = 16
        next(f)
        for line in f:
            line = line.strip().split("\t")
            cur_id = line[id_col]
            cur_lineage = line[lineage_col]

            if cur_id in id2index:
                lineages.append(cur_lineage)
                if cur_lineage not in sequences_per_lineage:
                    sequences_per_lineage[cur_lineage] = []
                sequences_per_lineage[cur_lineage].append(cur_id)
    
    return sequences, records, lineages, id2index, index2id, sequences_per_lineage

def sketch(input_pair):
    id, seq = input_pair
    mh = sourmash.MinHash(n=0, ksize=31, scaled=5, track_abundance=True)
    mh.add_sequence(seq, force=True)
    return (id, mh)

def compute_distance_batch(args):
        idx, index2id, sketches, num_sequences = args
        res = [idx]
        for j in range(idx+1, num_sequences):
            d = sketches[index2id[idx]].similarity(sketches[index2id[j]])
            res.append(1.0 - d)
        return res

def main():
    SELECTION_COST = 0.1
    MAX_FRACTION = 0.5

    MAX_ITERATIONS = 1_000_000
    NUM_CORES = 32
    LOGGING = True
    LOGGING_FREQUENCY = 300

    sequences, records, lineages, id2index, index2id, sequences_per_lineage = read_genomes()
    lineages_unique = sorted(list(set(lineages)))
    clusters = []
    for lineage in lineages:
        clusters.append(lineages_unique.index(lineage))
    clusters = np.array(clusters, dtype=np.int32)

    try:
        distances = np.load("_distances.npy")
        print("Loaded precomputed distances.", flush=True)
    except FileNotFoundError:
        print("Precomputed distances not found, computing from scratch.", flush=True)
        pairs = [(id, seq) for id, seq in sequences.items()]
        print("Sketching genomes...", flush=True)
        start = time.time()
        with Pool(NUM_CORES) as pool:
            sketches = pool.map(sketch, pairs)
        sketches = {id: mh for id, mh in sketches}
        print(f"Sketching completed in {time.time() - start:.4f}s.", flush=True)
        distances = np.zeros((len(sequences), len(sequences)), dtype=np.float64)

        indices = list(range(len(sequences)))[:-1]  # final index has no other indices for which to compute distances
        args_list = [(idx, index2id, sketches, len(sequences)) for idx in indices]
        print("Computing pairwise distances...", flush=True)
        start = time.time()
        with Pool(NUM_CORES) as pool:
            results = pool.map(compute_distance_batch, args_list)
        for res in results:
            idx = res[0]
            for j in range(1, len(res)):
                distances[idx, idx+j] = res[j]
                distances[idx+j, idx] = res[j]
        print(f"Pairwise distances computed in {time.time() - start:.4f}s.", flush=True)

    MAX_ITERATIONS = 950
    LOGGING_FREQUENCY = 100
    SEED = 12345
    MAX_FRACTION = 0.1
    
    S = Solution.generate_random_solution(
        distances=distances,
        clusters=clusters,
        selection_cost=SELECTION_COST,
        cost_per_cluster=0,
        max_fraction=MAX_FRACTION,
        seed=SEED
    )

    start = time.time()
    time_per_iteration, objectives = S.local_search(
        max_iterations=MAX_ITERATIONS,
        max_runtime=600,
        doubleswap_time_threshold=30.0,
        logging=LOGGING,
        logging_frequency=LOGGING_FREQUENCY,
        random_move_order=True, random_index_order=True,
    )
    end = time.time()
    print(f"Local search completed (old). Objective = {objectives[-1]}", flush=True)
    print(f"Total time elapsed: {end - start}s.", flush=True)
    print(f"Average iteration time: {np.mean(time_per_iteration[-200:])}s (std: {np.std(time_per_iteration[-200:], ddof=1)}s).", flush=True)
    print()
    
    S = Solution_shm.generate_random_solution(
        distances=distances,
        clusters=clusters,
        selection_cost=SELECTION_COST,
        cost_per_cluster=0,
        max_fraction=MAX_FRACTION,
        seed=SEED
    )
    print("Object initialized")
    start = time.time()
    time_per_iteration, objectives = S.local_search(
        num_processes = 16,
        max_iterations=MAX_ITERATIONS,
        max_runtime=600,
        doubleswap_time_threshold=60.0,
        logging=LOGGING,
        logging_frequency=LOGGING_FREQUENCY,
        random_move_order=True, random_index_order=True,
    )
    end = time.time()
    print(f"Local search completed (shm). Objective = {objectives[-1]}", flush=True)
    print(f"Total time elapsed: {end - start}s.", flush=True)
    print(f"Average iteration time: {np.mean(time_per_iteration[-200:])}s (std: {np.std(time_per_iteration[-200:], ddof=1)}s).", flush=True)
    print()


if __name__ == "__main__":
    main()