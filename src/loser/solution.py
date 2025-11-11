"""
TO-DO:
    - Potential problem with cluster labels that are not necessarily 0-indexed or consecutive!!
"""
import numpy as np
from scipy.spatial.distance import squareform
import itertools
import math
import multiprocessing.shared_memory as shm
from multiprocessing import Pool, Manager
import time
import traceback
from collections import deque

# This is to define the precision threshold for floating point comparisons
PRECISION_THRESHOLD = 1e-10
DISTANCE_DTYPE = np.float64
AUXILIARY_DISTANCE_DTYPE = np.float64

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

        self.calculate_objective()
        
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

    def local_search_sp(self,
                        max_iterations: int = 10_000, max_runtime: float = np.inf,
                        random_move_order: bool = True, random_index_order: bool = True, move_order: list = ["add", "swap", "doubleswap", "remove"],
                        dynamically_check: bool = False, max_move_queue_size: int = 1000, min_doubleswaps: int = 1, start_p: float = 0.25, decay_rate: float = 0.01,
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
        dynamically_check: bool
            If True, doubleswaps will be dynamically checked based on the recent moves,
            and will be omitted if not performed frequently enough.
            NOTE: If set to false, doubleswaps will always be checked
            and all moves are assigned equal probability.
        max_move_queue_size: int
            The maximum number of moves to keep track of in the recent moves queue.
        min_doubleswaps: int
            The minimum number of doubleswaps in the last max_move_queue_size moves
            before doubleswaps are omitted.
        start_p: float
            The starting probability for testing a doubleswap move.
            NOTE: This probability should be larger than 1/4 to ensure that
            doubleswaps are tested enough to conclude that they can be
            omitted.
        decay_rate: float
            The rate at which the probability for testing a doubleswap move decays.
            NOTE: This should be a small positive number, e.g. 0.01.
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
        if not isinstance(dynamically_check, bool):
            raise ValueError("dynamically_check must be a boolean value.")
        if not isinstance(max_move_queue_size, int) or max_move_queue_size < 1:
            raise ValueError("max_move_queue_size must be a positive integer.")
        if not isinstance(min_doubleswaps, int) or min_doubleswaps < 0:
            raise ValueError("min_doubleswaps must be a non-negative integer.")
        if not isinstance(start_p, float) or not (0 < start_p <= 1):
            raise ValueError("start_p must be a float between 0 (exclusive) and 1 (inclusive).")
        if not isinstance(decay_rate, float) or decay_rate < 0:
            raise ValueError("decay_rate must be a non-negative float.")
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
        # Initialize variables for dynamically checking doubleswaps
        if dynamically_check:
            recent_moves = deque(maxlen=max_move_queue_size)
            check_doubleswap = True
        else:
            check_doubleswap = False

        start_time = time.time()
        while iteration < max_iterations:
            current_iteration_time = time.time()
            objectives.append(self.objective)
            solution_changed = False
            if dynamically_check:
                move_generator = self.generate_moves_biased(
                    iteration=iteration,
                    random_move_order=random_move_order,
                    random_index_order=random_index_order,
                    order=move_order,
                    decay_rate=decay_rate,
                    start_p=start_p
                )
            else:
                move_generator = self.generate_moves(
                    random_move_order=random_move_order,
                    random_index_order=random_index_order,
                    order=move_order
                )
            
            move_counter = 0
            for move_type, move_content in move_generator:
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
                # Check if doubleswaps should be removed
                if check_doubleswap and dynamically_check:
                    recent_moves.append(move_type)
                    if len(recent_moves) >= max_move_queue_size:
                        num_doubleswaps = sum(1 for move in recent_moves if move == "doubleswap")
                        if num_doubleswaps < min_doubleswaps:
                            check_doubleswap = False
                            del recent_moves
                            move_order = [move for move in move_order if move != "doubleswap"]
                            if logging:
                                print(f"Disabled doubleswap moves after {iteration} iterations due to insufficient doubleswaps in the last {max_move_queue_size} moves.", flush=True)
            else:
                break

            if iteration % logging_frequency == 0 and logging:
                print(f"Iteration {iteration}: Objective = {self.objective:.10f}", flush=True)
                print(f"Average runtime last {logging_frequency} iterations: {np.mean(time_per_iteration[-logging_frequency:]):.6f} seconds", flush=True)

        return time_per_iteration, objectives

    def local_search_mp(self, 
                        max_iterations: int = 10_000, max_runtime: float = np.inf,
                        num_cores: int = 2,
                        random_move_order: bool = True, random_index_order: bool = True, move_order: list = ["add", "swap", "doubleswap", "remove"],
                        batch_size: int = 1000, max_batches: int = 32, 
                        runtime_switch: float = 10.0,
                        dynamically_check: bool = False, max_move_queue_size: int = 1000, min_doubleswaps: int = 1, start_p: float = 0.25, decay_rate: float = 0.01,
                        logging: bool = False, logging_frequency: int = 500,
                        ):
        """
        Perform local search to find a (local) optimal solution using an adaptive approach where
        the search switches between single-core and multi-core execution based on the runtime of iterations.

        Parameters:
        max_iterations: int
            The maximum number of iterations to perform.
        max_runtime: float
            The maximum runtime in seconds for the local search.
        num_cores: int
            The number of cores to use for parallel processing.
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
        dynamically_check: bool
            If True, doubleswaps will be dynamically checked based on the recent moves,
            and will be omitted if not performed frequently enough.
            NOTE: If set to false, doubleswaps will always be checked
            and all moves are assigned equal probability.
        max_move_queue_size: int
            The maximum number of moves to keep track of in the recent moves queue.
        min_doubleswaps: int
            The minimum number of doubleswaps in the last max_move_queue_size moves
            before doubleswaps are omitted.
        start_p: float
            The starting probability for testing a doubleswap move.
            NOTE: This probability should be larger than 1/4 to ensure that
            doubleswaps are test enough to conclude that they can be
            omitted.
        decay_rate: float
            The rate at which the probability for testing a doubleswap move decays.
            NOTE: This should be a small positive number, e.g. 0.01.
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
        if not isinstance(move_order, list):
            raise ValueError("move_order must be a list of move types.")
        else:
            if len(move_order) == 0:
                raise ValueError("move_order must contain at least one move type.")
            valid_moves = {"add", "swap", "doubleswap", "remove"}
            if len(set(move_order) - valid_moves) > 0:
                raise ValueError("move_order must contain only the following move types: add, swap, doubleswap, remove.")
        if not isinstance(dynamically_check, bool):
            raise ValueError("dynamically_check must be a boolean value.")
        if not isinstance(max_move_queue_size, int) or max_move_queue_size < 1:
            raise ValueError("max_move_queue_size must be a positive integer.")
        if not isinstance(min_doubleswaps, int) or min_doubleswaps < 0:
            raise ValueError("min_doubleswaps must be a non-negative integer.")
        if not isinstance(start_p, float) or not (0 < start_p <= 1):
            raise ValueError("start_p must be a float between 0 (exclusive) and 1 (inclusive).")
        if not isinstance(decay_rate, float) or decay_rate < 0:
            raise ValueError("decay_rate must be a non-negative float.")
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
        # Initialize variables for dynamically checking doubleswaps
        if dynamically_check:
            recent_moves = deque(maxlen=max_move_queue_size)
            check_doubleswap = True
        else:
            check_doubleswap = False
        # Initialize variables for multiprocessing
        run_in_multiprocessing = False

        # Ensure shm handles exist
        distances_shm = None
        clusters_shm = None
        closest_distances_intra_shm = None
        closest_points_intra_shm = None
        closest_distances_inter_shm = None
        closest_points_inter_shm = None

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

            # For the intra and inter distances, only copy them during iterations since they are updated during the local search
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
            closest_points_inter_shm = shm.SharedMemory(create=True, size=self.closest_points_inter.nbytes)
            shared_closest_points_inter = np.ndarray(self.closest_points_inter.shape, dtype=self.closest_points_inter.dtype, buffer=closest_points_inter_shm.buf)
            
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
                    
                    start_time = time.time()
                    while iteration < max_iterations:
                        current_iteration_time = time.time()
                        objectives.append(self.objective)
                        solution_changed = False
                        run_in_multiprocessing = False
                        if dynamically_check:
                            move_generator = self.generate_moves_biased(
                                iteration=iteration, 
                                random_move_order=random_move_order, 
                                random_index_order=random_index_order, 
                                order=move_order,
                                decay_rate=decay_rate, 
                                start_p=start_p
                            )
                        else:
                            move_generator = self.generate_moves(
                                random_move_order=random_move_order, 
                                random_index_order=random_index_order, 
                                order=move_order
                            )

                        move_counter = 0
                        for move_type, move_content in move_generator:
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

                            if move_counter % 1_000 == 0: #every 1000 moves, check if we should switch to multiprocessing or terminate
                                # Check if total runtime exceeds max_runtime
                                if time.time() - start_time > max_runtime:
                                    if logging:
                                        print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time}), stopping local search.", flush=True)
                                    return time_per_iteration, objectives
                                # Check if current iteration should switch to multiprocessing
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
                            np.copyto(shared_closest_points_inter, self.closest_points_inter)
                            
                            event.clear() #reset event for current iteration
                            results = [] #resets results for current iteration

                            num_solutions_tried = 0
                            # Try moves in batches
                            while True:
                                batches = []
                                num_this_loop = 0
                                cur_batch_time = time.time()
                                for _ in range(max_batches): #fill list with up to max_batches batches
                                    batch = []
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
                                        move_type, idxs_to_add, idxs_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters = results[0]
                                        break
                                    else:
                                        num_solutions_tried += num_this_loop
                                        if logging:
                                            print(f"Processed {num_solutions_tried} solutions (current batch took {time.time() - cur_batch_time:.2f}s), no improvement found yet.", flush=True)
                                    if time.time() - start_time > max_runtime:
                                        if logging:
                                            print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time}), stopping local search.", flush=True)
                                        return time_per_iteration, objectives
                                else: # No more tasks to process, break while loop
                                    break

                        time_per_iteration.append(time.time() - current_iteration_time)
                        if solution_changed: # If improvement is found, update solution
                            self.accept_move(idxs_to_add, idxs_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                            del idxs_to_add, idxs_to_remove #sanity check, should throw error if something weird happens
                            iteration += 1 #update iteration count
                            # Check if time exceeds allowed runtime
                            if time.time() - start_time > max_runtime:
                                if logging:
                                    print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time}), stopping local search.", flush=True)
                                return time_per_iteration, objectives
                            # Check if doubleswaps should be removed
                            if check_doubleswap and dynamically_check:
                                recent_moves.append(move_type)
                                if len(recent_moves) == max_move_queue_size:
                                    num_doubleswaps = sum(1 for move in recent_moves if move == "doubleswap")
                                    if num_doubleswaps < min_doubleswaps:
                                        check_doubleswap = False
                                        del recent_moves
                                        move_order = [move for move in move_order if move != "doubleswap"]
                                        if logging:
                                            print(f"Disabled doubleswap moves after {iteration} iterations due to insufficient doubleswaps in the last {max_move_queue_size} moves.", flush=True)

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
                try:
                    distances_shm.close()
                    distances_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for distances already unlinked, exiting as normal.", flush=True)
            if clusters_shm:
                try:
                    clusters_shm.close()
                    clusters_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for clusters already unlinked, exiting as normal.", flush=True)
            if closest_distances_intra_shm:
                try:
                    closest_distances_intra_shm.close()
                    closest_distances_intra_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for closest distances intra already unlinked, exiting as normal.", flush=True)
            if closest_points_intra_shm:
                try:
                    closest_points_intra_shm.close()
                    closest_points_intra_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for closest points intra already unlinked, exiting as normal.", flush=True)
            if closest_distances_inter_shm:
                try:
                    closest_distances_inter_shm.close()
                    closest_distances_inter_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for closest distances inter already unlinked, exiting as normal.", flush=True)
            if closest_points_inter_shm:
                try:
                    closest_points_inter_shm.close()
                    closest_points_inter_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for closest points inter already unlinked, exiting as normal.", flush=True)

        return time_per_iteration, objectives

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

    def generate_moves_biased(self, iteration, random_move_order: bool = True, random_index_order: bool = True, order=["add", "swap", "doubleswap", "remove"], decay_rate=0.01, start_p=0.5):
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

        probabilities = np.zeros(len(active_generators), dtype=np.float64)
        if "doubleswap" in active_generators:
            p_doubleswap = 1/len(active_generators) + (start_p - 1/len(active_generators)) * np.exp(-decay_rate * iteration)
            p_doubleswap = min(max(p_doubleswap, 0.0), 1.0)
            remaining = 1.0 - p_doubleswap
            doubleswap_index = active_generators.index("doubleswap")
            for i in range(len(active_generators)):
                if i != doubleswap_index:
                    probabilities[i] = remaining / (len(active_generators) - 1)
                else:
                    probabilities[doubleswap_index] = p_doubleswap
        else:
            probabilities = np.ones(len(active_generators), dtype=np.float64) / len(active_generators)
        


        # While there are active generators, yield from them until exhausted
        while active_generators:
            if random_move_order:
                selected_generator = self.random_state.choice(active_generators, p=probabilities)
            else:
                selected_generator = active_generators[0]
            # This try-except block allows to yield from generator, and if no more of the corresponding move, removes it from active generators
            try:
                yield selected_generator, next(generators[selected_generator])
            except StopIteration:
                active_generators.remove(selected_generator)
                probabilities = np.zeros(len(active_generators), dtype=np.float64)
                if "doubleswap" in active_generators:
                    if len(active_generators) > 1:
                        p_doubleswap = 1/len(active_generators) + (start_p - 1/len(active_generators)) * np.exp(-decay_rate * iteration)
                        p_doubleswap = min(max(p_doubleswap, 0.0), 1.0)  # Ensure p_doubleswap does not exceed 1
                        remaining = 1.0 - p_doubleswap
                        doubleswap_index = active_generators.index("doubleswap")
                        for i in range(len(active_generators)):
                            if i != doubleswap_index:
                                probabilities[i] = remaining / (len(active_generators) - 1)
                            else:
                                probabilities[doubleswap_index] = p_doubleswap
                    else:
                        probabilities = np.array([1], dtype=np.float64)
                else:
                    probabilities = np.ones(len(active_generators), dtype=np.float64) / len(active_generators)

class SolutionAverage(Solution):
    """
    A specialized version of the Solution class that calculates inter-cluster distances
    as the average of similarities between cluster representatives of different clusters.
    """
    def __init__(self, distances: np.ndarray, clusters: np.ndarray, selection=None, selection_cost: float = 1.0, cost_per_cluster: int = 0, softmax_beta: float = 0.0, seed=None):
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
        self.num_clusters = self.unique_clusters.shape[0]

        # Process initial representation to optimize for comparisons speed
        self.points_per_cluster = {cluster: set(np.where(self.clusters == cluster)[0]) for cluster in self.unique_clusters} #points in every cluster

        # Determine logsum factor by calculating max similarity between points of different labels
        # NOTE: This is slow as of now, but can potentially be sped up in the future
        self.beta = softmax_beta
        self.logsum_factor = -np.inf
        for i in range(self.clusters.shape[0]):
            for j in range(i):
                if self.clusters[i] != self.clusters[j]:
                    self.logsum_factor = max(self.logsum_factor, self.beta*(1.0 - get_distance(i, j, self.distances, self.num_points)))
        self.calculate_objective()

    def __eq__(self, other):
        """
        Check if two solutions are equal.
        NOTE: This purely checks if all attributes are equal, excluding the random state.
        NOTE: This is mostly duplicate code from parent class except for inter-cluster distances
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
        # Check if inter cluster numerators are equal
        try:
            if not np.allclose(self.distances_inter_numerator, other.distances_inter_numerator, atol=PRECISION_THRESHOLD):
                print("Inter cluster distances numerators are not equal.")
                return False
        except:
            print("Inter cluster distances numerators could not be compared.")
            return False
        # Check if inter cluster denominators are equal
        try:
            if not np.allclose(self.distances_inter_denominator, other.distances_inter_denominator, atol=PRECISION_THRESHOLD):
                print("Inter cluster distances denominators are not equal.")
                return False
        except:
            print("Inter cluster distances denominators could not be compared.")
            return False
        # Check if betas are equal
        if not math.isclose(self.beta, other.beta, rel_tol=PRECISION_THRESHOLD):
            print("Betas are not equal.")
            return False
        # Check if logsum factors are equal
        if not math.isclose(self.logsum_factor, other.logsum_factor, rel_tol=PRECISION_THRESHOLD):
            print("Logsum factors are not equal.")
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
        """
        Arrays below are used for determining 'average' similarity between selected points of cluster labels as follows:
        avg_sim = ( ∑s(x,y) ⋅ exp{β⋅s(x,y)} ) / ( ∑exp{β⋅s(x,y)} )
        NOTE: we use log-sum-exp trick by taking m := max( β⋅s(x,y) ∀(x,y) ) and 'subtracting' this from the exponents
        (i.e. dividing by exp{m} for both the numerator and denominator) in order to maintain numerical stability.
        NOTE: when β=0 this is equivalent to calculating the mean over all similarities and thus it is checked
        whether or not β is equal to 0.
        """
        self.distances_inter_numerator = np.zeros(int(self.unique_clusters.shape[0] * (self.unique_clusters.shape[0] - 1) / 2), dtype=AUXILIARY_DISTANCE_DTYPE)
        self.distances_inter_denominator = np.zeros(int(self.unique_clusters.shape[0] * (self.unique_clusters.shape[0] - 1) / 2), dtype=AUXILIARY_DISTANCE_DTYPE)

        is_feasible = self.determine_feasibility()
        if not is_feasible:
            self.feasible = False
            self.objective = np.inf
            print("The solution is infeasible, objective value is set to INF and the closest distances & points are not set.")
            return self.objective
        self.feasible = True

        # Calculate the objective value
        objective = 0.0
        objective_components = {
            "selection": 0.0,
            "intra-cluster": 0.0,
            "inter-cluster": 0.0,
        } #maintain components of objective since inter-cluster component needs to be recalibrated from time to time?
        # Selection cost
        for idx in np.where(self.selection)[0]:
            objective += self.cost_per_cluster[self.clusters[idx]]
            objective_components["selection"] += self.cost_per_cluster[self.clusters[idx]]
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
                objective_components["intra-cluster"] += cur_min
        # Inter cluster distance costs
        for cluster_1, cluster_2 in itertools.combinations(self.unique_clusters, 2):
            cur_numerator = 0.0
            cur_denominator = 0.0
            for point_1, point_2 in itertools.product(self.selection_per_cluster[cluster_1], self.selection_per_cluster[cluster_2]):
                cur_similarity = 1.0 - get_distance(point_1, point_2, self.distances, self.num_points)
                cur_numerator += cur_similarity * math.exp(cur_similarity * self.beta - self.logsum_factor)
                cur_denominator += math.exp(cur_similarity * self.beta - self.logsum_factor)
            self.distances_inter_numerator[get_index(cluster_1, cluster_2, self.num_clusters)] = AUXILIARY_DISTANCE_DTYPE(cur_numerator)
            self.distances_inter_denominator[get_index(cluster_1, cluster_2, self.num_clusters)] = AUXILIARY_DISTANCE_DTYPE(cur_denominator)
            objective += cur_numerator/cur_denominator
            objective_components["inter-cluster"] += cur_numerator/cur_denominator

        self.objective = objective
        self.objective_components = objective_components

        return objective

    @classmethod
    def generate_centroid_solution(cls, distances, clusters, selection_cost: float = 1.0, cost_per_cluster: int = 0, softmax_beta: float = 0.0, seed=None):
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

        return cls(distances, clusters, selection=selection, selection_cost=selection_cost, cost_per_cluster=cost_per_cluster, softmax_beta=softmax_beta, seed=seed)
    
    @classmethod
    def generate_random_solution(cls, distances, clusters, selection_cost: float = 1.0, cost_per_cluster: int = 0, max_fraction=0.1, softmax_beta: float = 0.0, seed=None):
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

        return cls(distances, clusters, selection, selection_cost=selection_cost, cost_per_cluster=cost_per_cluster, softmax_beta=softmax_beta, seed=random_state)

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
            NOTE: If local_search is True, the returned value can be np.inf if
            the candidate objective is worse than the current objective based
            on intra distances.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
            NOTE: new_closest_point will always be idx_to_add.
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, new_numerator, new_denominator)]
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

        # Inter-cluster distances for other clusters
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                old_numerator = get_distance(cluster, other_cluster, self.distances_inter_numerator, self.num_clusters)
                old_denominator = get_distance(cluster, other_cluster, self.distances_inter_denominator, self.num_clusters)
                new_numerator = old_numerator
                new_denominator = old_denominator
                for idx in self.selection_per_cluster[other_cluster]:
                    cur_similarity = 1.0 - get_distance(idx, idx_to_add, self.distances, self.num_points)
                    new_numerator += cur_similarity * math.exp(cur_similarity * self.beta - self.logsum_factor)
                    new_denominator += math.exp(cur_similarity * self.beta - self.logsum_factor)
                candidate_objective += (new_numerator/new_denominator) - (old_numerator/old_denominator) #change in objective
                add_for_other_clusters.append((other_cluster, new_numerator, new_denominator))
                
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
            The objective value of the solution after the swap.
            NOTE: if the addition does not improve the objective, -1 is returned.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
            NOTE: if the addition does not improve the objective, -1 is returned.
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, new_numerator, new_denominator)]
            NOTE: if the addition does not improve the objective, -1 is returned.
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
        candidate_objective = self.objective + (num_to_add - 1) * self.cost_per_cluster[cluster]

        # Calculate intra-cluster distances
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
            else: #point to be removed is not closest, check if one of newly added points is closer
                cur_dists = [(get_distance(idx, idx_to_add, self.distances, self.num_points), idx_to_add) for idx_to_add in idxs_to_add]
                cur_dist, idx_to_add = min(cur_dists, key=lambda x: x[0])
                if cur_dist < cur_closest_distance:
                    candidate_objective += cur_dist - cur_closest_distance
                    add_within_cluster.append((idx, idx_to_add, cur_dist))

        # Calculate inter-cluster distances for all other clusters
        add_for_other_clusters = [] 
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                old_numerator = get_distance(cluster, other_cluster, self.distances_inter_numerator, self.num_clusters)
                old_denominator = get_distance(cluster, other_cluster, self.distances_inter_denominator, self.num_clusters)
                new_numerator = old_numerator
                new_denominator = old_denominator
                for idx in self.selection_per_cluster[other_cluster]:
                    # Similarities for point(s) to add
                    for idx_to_add in idxs_to_add:
                        cur_similarity = 1.0 - get_distance(idx, idx_to_add, self.distances, self.num_points)
                        new_numerator += cur_similarity * math.exp(cur_similarity * self.beta - self.logsum_factor)
                        new_denominator += math.exp(cur_similarity * self.beta - self.logsum_factor)
                    # Similarities for point to remove
                    cur_similarity = 1.0 - get_distance(idx, idx_to_remove, self.distances, self.num_points)
                    new_numerator -= cur_similarity * math.exp(cur_similarity * self.beta - self.logsum_factor)
                    new_denominator -= math.exp(cur_similarity * self.beta - self.logsum_factor)
                candidate_objective += (new_numerator/new_denominator) - (old_numerator/old_denominator) #change in objective
                add_for_other_clusters.append((other_cluster, new_numerator, new_denominator))

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
            The objective value of the solution after the addition.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, new_numerator, new_denominator)]
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
        add_for_other_clusters = [] 
        for other_cluster in self.unique_clusters:
            if other_cluster != cluster:
                old_numerator = get_distance(cluster, other_cluster, self.distances_inter_numerator, self.num_clusters)
                old_denominator = get_distance(cluster, other_cluster, self.distances_inter_denominator, self.num_clusters)
                new_numerator = old_numerator
                new_denominator = old_denominator
                for idx in self.selection_per_cluster[other_cluster]:
                    cur_similarity = 1.0 - get_distance(idx, idx_to_remove, self.distances, self.num_points)
                    new_numerator -= cur_similarity * math.exp(cur_similarity * self.beta - self.logsum_factor)
                    new_denominator -= math.exp(cur_similarity * self.beta - self.logsum_factor)
                candidate_objective += (new_numerator/new_denominator) - (old_numerator/old_denominator)
                add_for_other_clusters.append((other_cluster, new_numerator, new_denominator))

        # NOTE: Intra-cluster distances can only increase when removing a point, so when doing local search we can exit here if objective is worse
        if candidate_objective > self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD and local_search:
            return np.inf, None, None

        # Calculate intra-cluster distances
        add_within_cluster = []
        for idx in new_nonselection:
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
            Structure: [(index_other_cluster, new_numerator, new_denominator)]
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
        # Updating inter-cluster numerators and denominators
        for other_cluster, new_numerator, new_denominator in add_for_other_clusters:
            self.distances_inter_numerator[get_index(cluster, other_cluster, self.num_clusters)] = new_numerator
            self.distances_inter_denominator[get_index(cluster, other_cluster, self.num_clusters)] = new_denominator
        
        self.objective = candidate_objective

    def local_search_mp(self, 
                        max_iterations: int = 10_000, max_runtime: float = np.inf,
                        num_cores: int = 2,
                        random_move_order: bool = True, random_index_order: bool = True, move_order: list = ["add", "swap", "doubleswap", "remove"],
                        batch_size: int = 1000, max_batches: int = 32, 
                        runtime_switch: float = 10.0,
                        dynamically_check: bool = False, max_move_queue_size: int = 1000, min_doubleswaps: int = 1, start_p: float = 0.25, decay_rate: float = 0.01,
                        logging: bool = False, logging_frequency: int = 500,
                        ):
        """
        Perform local search to find a (local) optimal solution using an adaptive approach where
        the search switches between single-core and multi-core execution based on the runtime of iterations.

        Parameters:
        max_iterations: int
            The maximum number of iterations to perform.
        max_runtime: float
            The maximum runtime in seconds for the local search.
        num_cores: int
            The number of cores to use for parallel processing.
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
        dynamically_check: bool
            If True, doubleswaps will be dynamically checked based on the recent moves,
            and will be omitted if not performed frequently enough.
            NOTE: If set to false, doubleswaps will always be checked
            and all moves are assigned equal probability.
        max_move_queue_size: int
            The maximum number of moves to keep track of in the recent moves queue.
        min_doubleswaps: int
            The minimum number of doubleswaps in the last max_move_queue_size moves
            before doubleswaps are omitted.
        start_p: float
            The starting probability for testing a doubleswap move.
            NOTE: This probability should be larger than 1/4 to ensure that
            doubleswaps are test enough to conclude that they can be
            omitted.
        decay_rate: float
            The rate at which the probability for testing a doubleswap move decays.
            NOTE: This should be a small positive number, e.g. 0.01.
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
        if not isinstance(move_order, list):
            raise ValueError("move_order must be a list of move types.")
        else:
            if len(move_order) == 0:
                raise ValueError("move_order must contain at least one move type.")
            valid_moves = {"add", "swap", "doubleswap", "remove"}
            if len(set(move_order) - valid_moves) > 0:
                raise ValueError("move_order must contain only the following move types: add, swap, doubleswap, remove.")
        if not isinstance(dynamically_check, bool):
            raise ValueError("dynamically_check must be a boolean value.")
        if not isinstance(max_move_queue_size, int) or max_move_queue_size < 1:
            raise ValueError("max_move_queue_size must be a positive integer.")
        if not isinstance(min_doubleswaps, int) or min_doubleswaps < 0:
            raise ValueError("min_doubleswaps must be a non-negative integer.")
        if not isinstance(start_p, float) or not (0 < start_p <= 1):
            raise ValueError("start_p must be a float between 0 (exclusive) and 1 (inclusive).")
        if not isinstance(decay_rate, float) or decay_rate < 0:
            raise ValueError("decay_rate must be a non-negative float.")
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
        # Initialize variables for dynamically checking doubleswaps
        if dynamically_check:
            recent_moves = deque(maxlen=max_move_queue_size)
            check_doubleswap = True
        else:
            check_doubleswap = False
        # Initialize variables for multiprocessing
        run_in_multiprocessing = False

        # Ensure shm handles exist
        distances_shm = None
        clusters_shm = None
        closest_distances_intra_shm = None
        closest_points_intra_shm = None
        distances_inter_numerator_shm = None
        distances_inter_denominator_shm = None

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

            # For the intra and inter distances, only copy them during iterations since they are updated during the local search
            # Copy closest_distances_intra to shared memory
            closest_distances_intra_shm = shm.SharedMemory(create=True, size=self.closest_distances_intra.nbytes)
            shared_closest_distances_intra = np.ndarray(self.closest_distances_intra.shape, dtype=self.closest_distances_intra.dtype, buffer=closest_distances_intra_shm.buf)
            # Copy closest_points_intra to shared memory
            closest_points_intra_shm = shm.SharedMemory(create=True, size=self.closest_points_intra.nbytes)
            shared_closest_points_intra = np.ndarray(self.closest_points_intra.shape, dtype=self.closest_points_intra.dtype, buffer=closest_points_intra_shm.buf)
            # Copy distances_inter_numerator to shared memory
            distances_inter_numerator_shm = shm.SharedMemory(create=True, size=self.distances_inter_numerator.nbytes)
            shared_distances_inter_numerator = np.ndarray(self.distances_inter_numerator.shape, dtype=self.distances_inter_numerator.dtype, buffer=distances_inter_numerator_shm.buf)
            # Copy distances_inter_denominator to shared memory
            distances_inter_denominator_shm = shm.SharedMemory(create=True, size=self.distances_inter_denominator.nbytes)
            shared_distances_inter_denominator = np.ndarray(self.distances_inter_denominator.shape, dtype=self.distances_inter_denominator.dtype, buffer=distances_inter_denominator_shm.buf)
            
            with Manager() as manager:
                event = manager.Event() #this is used to signal when tasks should be stopped
                results = manager.list() #this is used to store an improvement is one is found

                with Pool(
                    processes=num_cores,
                    initializer=init_worker_avg,
                    initargs=(
                        distances_shm.name, shared_distances.shape,
                        clusters_shm.name, shared_clusters.shape,
                        closest_distances_intra_shm.name, shared_closest_distances_intra.shape,
                        closest_points_intra_shm.name, shared_closest_points_intra.shape,
                        distances_inter_numerator_shm.name, shared_distances_inter_numerator.shape,
                        distances_inter_denominator_shm.name, shared_distances_inter_denominator.shape,
                        self.unique_clusters, self.cost_per_cluster, self.num_points, self.num_clusters, self.beta, self.logsum_factor,
                    ),
                ) as pool:
                    
                    start_time = time.time()
                    while iteration < max_iterations:
                        current_iteration_time = time.time()
                        objectives.append(self.objective)
                        solution_changed = False
                        run_in_multiprocessing = False
                        if dynamically_check:
                            move_generator = self.generate_moves_biased(
                                iteration=iteration, 
                                random_move_order=random_move_order, 
                                random_index_order=random_index_order, 
                                order=move_order,
                                decay_rate=decay_rate, 
                                start_p=start_p
                            )
                        else:
                            move_generator = self.generate_moves(
                                random_move_order=random_move_order, 
                                random_index_order=random_index_order, 
                                order=move_order
                            )

                        move_counter = 0
                        for move_type, move_content in move_generator:
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

                            if move_counter % 1_000 == 0: #every 1000 moves, check if we should switch to multiprocessing or terminate
                                # Check if total runtime exceeds max_runtime
                                if time.time() - start_time > max_runtime:
                                    if logging:
                                        print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time}), stopping local search.", flush=True)
                                    return time_per_iteration, objectives
                                # Check if current iteration should switch to multiprocessing
                                if time.time() - current_iteration_time > runtime_switch:
                                    if logging:
                                        print(f"Iteration {iteration+1} is taking longer than {runtime_switch} seconds, switching to multiprocessing.", flush=True)
                                    run_in_multiprocessing = True
                                    break #break out of singleprocessing

                        if run_in_multiprocessing: #If switching to multiprocessing
                            # Start by updating shared memory arrays
                            np.copyto(shared_closest_distances_intra, self.closest_distances_intra)
                            np.copyto(shared_closest_points_intra, self.closest_points_intra)
                            np.copyto(shared_distances_inter_numerator, self.distances_inter_numerator)
                            np.copyto(shared_distances_inter_denominator, self.distances_inter_denominator)
                            
                            event.clear() #reset event for current iteration
                            results = [] #resets results for current iteration

                            num_solutions_tried = 0
                            # Try moves in batches
                            while True:
                                batches = []
                                num_this_loop = 0
                                cur_batch_time = time.time()
                                for _ in range(max_batches): #fill list with up to max_batches batches
                                    batch = []
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
                                            process_batch_avg,
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
                                        move_type, idxs_to_add, idxs_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters = results[0]
                                        break
                                    else:
                                        num_solutions_tried += num_this_loop
                                        if logging:
                                            print(f"Processed {num_solutions_tried} solutions (current batch took {time.time() - cur_batch_time:.2f}s), no improvement found yet.", flush=True)
                                    if time.time() - start_time > max_runtime:
                                        if logging:
                                            print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time}), stopping local search.", flush=True)
                                        return time_per_iteration, objectives
                                else: # No more tasks to process, break while loop
                                    break

                        time_per_iteration.append(time.time() - current_iteration_time)
                        if solution_changed: # If improvement is found, update solution
                            self.accept_move(idxs_to_add, idxs_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters)
                            del idxs_to_add, idxs_to_remove #sanity check, should throw error if something weird happens
                            iteration += 1 #update iteration count
                            # Check if time exceeds allowed runtime
                            if time.time() - start_time > max_runtime:
                                if logging:
                                    print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time}), stopping local search.", flush=True)
                                return time_per_iteration, objectives
                            # Check if doubleswaps should be removed
                            if check_doubleswap and dynamically_check:
                                recent_moves.append(move_type)
                                if len(recent_moves) == max_move_queue_size:
                                    num_doubleswaps = sum(1 for move in recent_moves if move == "doubleswap")
                                    if num_doubleswaps < min_doubleswaps:
                                        check_doubleswap = False
                                        del recent_moves
                                        move_order = [move for move in move_order if move != "doubleswap"]
                                        if logging:
                                            print(f"Disabled doubleswap moves after {iteration} iterations due to insufficient doubleswaps in the last {max_move_queue_size} moves.", flush=True)

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
                try:
                    distances_shm.close()
                    distances_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for distances already unlinked, exiting as normal.", flush=True)
            if clusters_shm:
                try:
                    clusters_shm.close()
                    clusters_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for clusters already unlinked, exiting as normal.", flush=True)
            if closest_distances_intra_shm:
                try:
                    closest_distances_intra_shm.close()
                    closest_distances_intra_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for closest distances intra already unlinked, exiting as normal.", flush=True)
            if closest_points_intra_shm:
                try:
                    closest_points_intra_shm.close()
                    closest_points_intra_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for closest points intra already unlinked, exiting as normal.", flush=True)
            if distances_inter_numerator_shm:
                try:
                    distances_inter_numerator_shm.close()
                    distances_inter_numerator_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for distances inter numerator already unlinked, exiting as normal.", flush=True)
            if distances_inter_denominator_shm:
                try:
                    distances_inter_denominator_shm.close()
                    distances_inter_denominator_shm.unlink()
                except FileNotFoundError:
                    print("Shared memory for distances inter denominator already unlinked, exiting as normal.", flush=True)

        return time_per_iteration, objectives
                                
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

    Returns:
    --------
    candidate_objective: float
        The objective value of the solution after the addition.
        NOTE: if the addition does not improve the objective, -1 is returned.
    add_within_cluster: list of tuples
        The changes to be made within the cluster of the added point.
        Structure: [(index_to_change, new_closest_point, new_distance)]
        NOTE: new_closest_point will always be idx_to_add.
        NOTE: if the addition does not improve the objective, -1 is returned.
    add_for_other_clusters: list of tuples
        The changes to be made for other clusters.
        Structure: [(index_other_cluster, (point_in_this_cluster, point_in_other_cluster), new_distance)]
        NOTE: point_in_this_cluster will always be idx_to_add.
        NOTE: if the addition does not improve the objective, -1 is returned.
    """
    cluster = _clusters[idx_to_add]

    # Calculate selection cost
    candidate_objective = objective + _cost_per_cluster[cluster] # cost for adding the point
        
    # Calculate intra-cluster distances
    add_within_cluster = [] #this stores changes that have to be made if the objective improves
    for idx in nonselection:
        cur_dist = get_distance(idx, idx_to_add, _distances, _num_points) #distance to current point (idx)
        if cur_dist < _closest_distances_intra[idx]:
            candidate_objective += cur_dist - _closest_distances_intra[idx]
            add_within_cluster.append((idx, idx_to_add, cur_dist))

    # NOTE: Inter-cluster distances can only increase when adding a point, so when doing local search we can exit here if objective is worse
    if candidate_objective > objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
        return -1, -1, -1

    # Calculate inter-cluster distances for other clusters
    add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
    for other_cluster in _unique_clusters:
        if other_cluster != cluster:
            cur_max = _closest_distances_inter[cluster, other_cluster]
            cur_idx = -1
            for idx in selection_per_cluster[other_cluster]:
                cur_similarity = 1.0 - get_distance(idx, idx_to_add, _distances, _num_points)
                if cur_similarity > cur_max:
                    cur_max = cur_similarity
                    cur_idx = idx
            if cur_idx > -1:
                candidate_objective += cur_max - _closest_distances_inter[cluster, other_cluster]
                add_for_other_clusters.append((other_cluster, (idx_to_add, cur_idx), cur_max))

    if candidate_objective < objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
        return candidate_objective, add_within_cluster, add_for_other_clusters
    else:
        return -1, -1, -1
        
def evaluate_add_mp_avg(
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
        NOTE: this version is specifically intended for solution objects that
        use the "average" costs for inter-cluster distances!

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

        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the addition.
            NOTE: if the addition does not improve the objective, -1 is returned.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
            NOTE: new_closest_point will always be idx_to_add.
            NOTE: if the addition does not improve the objective, -1 is returned.
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, new_numerator, new_denominator)]
            NOTE: if the addition does not improve the objective, -1 is returned.
        """
        cluster = _clusters[idx_to_add]

        # Calculate selection cost
        candidate_objective = objective + _cost_per_cluster[cluster] # cost for adding the point
        
        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for idx in nonselection:
            cur_dist = get_distance(idx, idx_to_add, _distances, _num_points) # distance to current point (idx)
            if cur_dist < _closest_distances_intra[idx]:
                candidate_objective += cur_dist - _closest_distances_intra[idx]
                add_within_cluster.append((idx, idx_to_add, cur_dist))

        # NOTE: Inter-cluster distances can only increase when adding a point, so when doing local search we can exit here if objective is worse
        if candidate_objective > objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
            return -1, -1, -1
        
        # Inter-cluster distances for other clusters
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        for other_cluster in _unique_clusters:
            if other_cluster != cluster:
                old_numerator = get_distance(cluster, other_cluster, _distances_inter_numerator, _num_clusters)
                old_denominator = get_distance(cluster, other_cluster, _distances_inter_denominator, _num_clusters)
                new_numerator = old_numerator
                new_denominator = old_denominator
                for idx in selection_per_cluster[other_cluster]:
                    cur_similarity = 1.0 - get_distance(idx, idx_to_add, _distances, _num_points)
                    new_numerator += cur_similarity * math.exp(cur_similarity * _beta - _logsum_factor)
                    new_denominator += math.exp(cur_similarity * _beta - _logsum_factor)
                candidate_objective += (new_numerator/new_denominator) - (old_numerator/old_denominator)
                add_for_other_clusters.append((other_cluster, new_numerator, new_denominator))

        if candidate_objective < objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
            return candidate_objective, add_within_cluster, add_for_other_clusters
        else:
            return -1, -1, -1

def evaluate_swap_mp(
        idxs_to_add: list, idx_to_remove: int, objective: float,
        selection_per_cluster: dict, nonselection: set):
    """
    Evaluates the effect of swapping a selected point for a/multiple unselected point(s)
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

    Returns:
    --------
    candidate_objective: float
        The objective value of the solution after the swap.
        NOTE: if the swap does not improve the objective, -1 is returned.
    add_within_cluster: list of tuples
        The changes to be made within the cluster of the added point.
        Structure: [(index_to_change, new_closest_point, new_distance)]
        NOTE: if the swap does not improve the objective, -1 is returned.
    add_for_other_clusters: list of tuples
        The changes to be made for other clusters.
        Structure: [(index_other_cluster, (point_in_this_cluster, point_in_other_cluster), new_distance)]
        NOTE: if the swap does not improve the objective, -1 is returned.
    """
    try:
        num_to_add = len(idxs_to_add)
    except TypeError:
        num_to_add = 1
        idxs_to_add = [idxs_to_add]
    cluster = _clusters[idx_to_remove]

    # Generate pool of alternative points to compare to
    new_selection = set(selection_per_cluster[cluster])
    for idx in idxs_to_add:
        new_selection.add(idx)
    new_selection.remove(idx_to_remove)
    new_nonselection = set(nonselection)
    new_nonselection.add(idx_to_remove)

    # Calculate selection cost
    candidate_objective = objective + (num_to_add - 1) * _cost_per_cluster[cluster] #cost for swapping points

    # Calculate intra-cluster distances
    add_within_cluster = [] #this stores changes that have to be made if the objective improves
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

    # Calculate inter-cluster distances for all other clusters
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
                            cur_closest_pair = (other_idx, idx)
            else: #point to be removed is not closest, check if one of newly added points is closer
                for idx in selection_per_cluster[other_cluster]:
                    cur_similarities = [(1.0 - get_distance(idx, idx_to_add, _distances, _num_points), idx_to_add) for idx_to_add in idxs_to_add]
                    cur_similarity, idx_to_add = max(cur_similarities, key = lambda x: x[0])
                    if cur_similarity > cur_closest_similarity:
                        cur_closest_similarity = cur_similarity
                        cur_closest_pair = (idx_to_add, idx)
            if cur_closest_pair[0] > -1:
                candidate_objective += cur_closest_similarity - _closest_distances_inter[cluster, other_cluster]
                add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))

    if candidate_objective < objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
        return candidate_objective, add_within_cluster, add_for_other_clusters
    else:
        return -1, -1, -1

def evaluate_swap_mp_avg(
        idxs_to_add, idx_to_remove: int, objective: float,
        selection_per_cluster: dict, nonselection: set):
    """
    Evaluates the effect of swapping a selected point for a/multiple unselected point(s)
    in the solution.
    NOTE: this function relies on shared memory, as well as existing variables that
    have to be initialized (those starting with an underscore) when spawning a worker
    process!
    NOTE: in the current implementation, there is no check for feasibility, so it is assumed
    that the swap can be performed without violating any constraints!
    NOTE: this version is specifically intended for solution objects that
    use the "average" costs for inter-cluster distances!
    
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

    Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the swap.
            NOTE: if the swap does not improve the objective, -1 is returned.
        add_within_cluster: list of tuples
            The changes to be made within the cluster of the added point.
            Structure: [(index_to_change, new_closest_point, new_distance)]
            NOTE: if the swap does not improve the objective, -1 is returned.
        add_for_other_clusters: list of tuples
            The changes to be made for other clusters.
            Structure: [(index_other_cluster, new_numerator, new_denominator)]
            NOTE: if the swap does not improve the objective, -1 is returned.
    """
    try:
        num_to_add = len(idxs_to_add)
    except TypeError:
        num_to_add = 1
        idxs_to_add = [idxs_to_add]
    cluster = _clusters[idx_to_remove]

    # Generate pool of alternative points to compare to
    new_selection = set(selection_per_cluster[cluster])
    for idx in idxs_to_add:
        new_selection.add(idx)
    new_selection.remove(idx_to_remove)
    new_nonselection = set(nonselection)
    new_nonselection.add(idx_to_remove)

    # Calculate selection cost
    candidate_objective = objective + (num_to_add - 1) * _cost_per_cluster[cluster] # cost for adding and removing

    # Calculate intra-cluster distances
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

    # Calculate inter-cluster distances for all other clusters
    add_for_other_clusters = []
    for other_cluster in _unique_clusters:
        if other_cluster != cluster:
            old_numerator = get_distance(cluster, other_cluster, _distances_inter_numerator, _num_clusters)
            old_denominator = get_distance(cluster, other_cluster, _distances_inter_denominator, _num_clusters)
            new_numerator = old_numerator
            new_denominator = old_denominator
            for idx in selection_per_cluster[other_cluster]:
                # Similarities for point(s) to add
                for idx_to_add in idxs_to_add:
                    cur_similarity = 1.0 - get_distance(idx, idx_to_add, _distances, _num_points)
                    new_numerator += cur_similarity * math.exp(cur_similarity * _beta - _logsum_factor)
                    new_denominator += math.exp(cur_similarity * _beta - _logsum_factor)
                # Similarities for point to remove
                cur_similarity = 1.0 - get_distance(idx, idx_to_remove, _distances, _num_points)
                new_numerator -= cur_similarity * math.exp(cur_similarity * _beta - _logsum_factor)
                new_denominator -= math.exp(cur_similarity * _beta - _logsum_factor)
            candidate_objective += (new_numerator/new_denominator) - (old_numerator/old_denominator)
            add_for_other_clusters.append((other_cluster, new_numerator, new_denominator))

    if candidate_objective < objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
        return candidate_objective, add_within_cluster, add_for_other_clusters
    else:
        return -1, -1, -1

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

    Returns:
    --------
    candidate_objective: float
        The objective value of the solution after the removal.
        NOTE: if the removal does not improve the objective, -1 is returned.
    add_within_cluster: list of tuples
        The changes to be made within the cluster of the removed point.
        Structure: [(index_to_change, new_closest_point, new_distance)]
        NOTE: if the removal does not improve the objective, -1 is returned.
    add_for_other_clusters: list of tuples
        The changes to be made for other clusters.
        Structure: [(index_other_cluster, (point_in_this_cluster, point_in_other_cluster), new_distance)]
        NOTE: if the removal does not improve the objective, -1 is returned.
    """
    cluster = _clusters[idx_to_remove]

    #Generate pool of alternative points to compare to
    new_selection = set(selection_per_cluster[cluster])
    new_selection.remove(idx_to_remove)
    new_nonselection = set(nonselection)
    new_nonselection.add(idx_to_remove)

    # Calculate selection cost
    candidate_objective = objective - _cost_per_cluster[cluster] # cost for removing the point from the cluster

    # Calculate inter-cluster distances for all other clusters
    # NOTE: Intra-cluster distances can only increase when removing a point, Thus if inter-cluster distances
    # increase, we can exit early.
    add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
    for other_cluster in _unique_clusters:
        if other_cluster != cluster:
            cur_closest_similarity = _closest_distances_inter[cluster, other_cluster]
            cur_closest_point = _closest_points_inter[cluster, other_cluster]
            cur_closest_pair = (-1, -1)
            if cur_closest_point == idx_to_remove:
                cur_closest_similarity = -np.inf
                for idx in selection_per_cluster[other_cluster]:
                    for other_idx in new_selection:
                        cur_similarity = 1.0 - get_distance(idx, other_idx, _distances, _num_points)
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            cur_closest_pair = (other_idx, idx)
                candidate_objective += cur_closest_similarity - _closest_distances_inter[cluster, other_cluster]
                add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
    
    # NOTE: Intra-cluster distances can only increase when removing a point, so when doing local search we can exit here if objective is worse
    if candidate_objective > objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
        return -1, -1, -1
    
    # Calculate intra-cluster distances
    add_within_cluster = [] #this stores changes that have to be made if the objective improves
    for idx in new_nonselection:
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

def evaluate_remove_mp_avg(
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
    NOTE: this version is specifically intended for solution objects that
    use the "average" costs for inter-cluster distances!

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
        
    Returns:
    --------
    candidate_objective: float
        The objective value of the solution after the removal.
        NOTE: if the removal does not improve the objective, -1 is returned.
    add_within_cluster: list of tuples
        The changes to be made within the cluster of the removed point.
        Structure: [(index_to_change, new_closest_point, new_distance)]
        NOTE: if the removal does not improve the objective, -1 is returned.
    add_for_other_clusters: list of tuples
        The changes to be made for other clusters.
        Structure: [(index_other_cluster, new_numerator, new_denominator)]
        NOTE: if the removal does not improve the objective, -1 is returned.
    """
    cluster = _clusters[idx_to_remove]

    # Generate pool of alternative points to compare to
    new_selection = set(selection_per_cluster[cluster])
    new_selection.remove(idx_to_remove)
    new_nonselection = set(nonselection)
    new_nonselection.add(idx_to_remove)

    # Calculate selection cost
    candidate_objective = objective - _cost_per_cluster[cluster] # cost for removing the point from the cluster

    # Calculate inter-cluster distances for all other clusters
    # NOTE: Intra-cluster distances can only increase when removing a point, Thus if inter-cluster distances
    # increase, we can exit early.
    add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
    for other_cluster in _unique_clusters:
        if other_cluster != cluster:
            old_numerator = get_distance(cluster, other_cluster, _distances_inter_numerator, _num_clusters)
            old_denominator = get_distance(cluster, other_cluster, _distances_inter_denominator, _num_clusters)
            new_numerator = old_numerator
            new_denominator = old_denominator
            for idx in selection_per_cluster[other_cluster]:
                cur_similarity = 1.0 - get_distance(idx, idx_to_remove, _distances, _num_points)
                new_numerator -= cur_similarity * math.exp(cur_similarity * _beta - _logsum_factor)
                new_denominator -= math.exp(cur_similarity * _beta - _logsum_factor)
            candidate_objective += (new_numerator/new_denominator) - (old_numerator/old_denominator)
            add_for_other_clusters.append((other_cluster, new_numerator, new_denominator))
    
    # NOTE: Intra-cluster distances can only increase when removing a point, so when doing local search we can exit here if objective is worse
    if candidate_objective > objective and np.abs(candidate_objective - objective) > PRECISION_THRESHOLD:
        return -1, -1, -1
    
    # Calculate intra-cluster distances
    add_within_cluster = [] #this stores changes that have to be made if the objective improves
    for idx in new_nonselection:
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
    _distances = np.ndarray(distances_shape, dtype=DISTANCE_DTYPE, buffer=_distances_shm.buf)
    global _clusters_shm, _clusters
    _clusters_shm = shm.SharedMemory(name=clusters_name)
    _clusters = np.ndarray(clusters_shape, dtype=np.int64, buffer=_clusters_shm.buf)
    global _closest_distances_intra_shm, _closest_distances_intra
    _closest_distances_intra_shm = shm.SharedMemory(name=closest_distances_intra_name)
    _closest_distances_intra = np.ndarray(closest_distances_intra_shape, dtype=DISTANCE_DTYPE, buffer=_closest_distances_intra_shm.buf)
    global _closest_points_intra_shm, _closest_points_intra
    _closest_points_intra_shm = shm.SharedMemory(name=closest_points_intra_name)
    _closest_points_intra = np.ndarray(closest_points_intra_shape, dtype=np.int32, buffer=_closest_points_intra_shm.buf)
    global _closest_distances_inter_shm, _closest_distances_inter
    _closest_distances_inter_shm = shm.SharedMemory(name=closest_distances_inter_name)
    _closest_distances_inter = np.ndarray(closest_distances_inter_shape, dtype=DISTANCE_DTYPE, buffer=_closest_distances_inter_shm.buf)
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

def init_worker_avg(
        distances_name, distances_shape, 
        clusters_name, clusters_shape,
        closest_distances_intra_name, closest_distances_intra_shape, 
        closest_points_intra_name, closest_points_intra_shape,
        distances_inter_numerator_name, distances_inter_numerator_shape,
        distances_inter_denominator_name, distances_inter_denominator_shape,
        unique_clusters, cost_per_cluster, num_points, num_clusters, beta, logsum_factor):
    """
    Initializes a worker for multiprocessing by setting up shared memory
    for the distances, clusters, closest distances and points.
    NOTE: this version is specifically intended for solution objects that
    use the "average" costs for inter-cluster distances!

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
    distances_inter_numerator_name: str
        Name of the shared memory segment for inter-cluster distances numerator.
    distances_inter_numerator_shape: tuple
        Shape of the inter-cluster distances numerator array.
    distances_inter_denominator_name: str
        Name of the shared memory segment for inter-cluster distances denominator.
    distances_inter_denominator_shape: tuple
        Shape of the inter-cluster distances denominator array.
    unique_clusters: np.ndarray
        Array of unique cluster indices.
    cost_per_cluster: np.ndarray
        Costs associated with selecting a point from a cluster.
        NOTE: This cost may be different for each cluster.
    num_points: int
        Total number of points in the dataset.
    num_clusters: int
        Total number of clusters in the dataset.
    beta: float
        Beta parameter for the average inter-cluster distance calculation.
    logsum_factor: float
        Logarithmic factor for the average inter-cluster distance calculation.
    """
    import numpy as np
    import multiprocessing.shared_memory as shm
    import atexit

    global _distances_shm, _distances
    _distances_shm = shm.SharedMemory(name=distances_name)
    _distances = np.ndarray(distances_shape, dtype=DISTANCE_DTYPE, buffer=_distances_shm.buf)
    global _clusters_shm, _clusters
    _clusters_shm = shm.SharedMemory(name=clusters_name)
    _clusters = np.ndarray(clusters_shape, dtype=np.int64, buffer=_clusters_shm.buf)
    global _closest_distances_intra_shm, _closest_distances_intra
    _closest_distances_intra_shm = shm.SharedMemory(name=closest_distances_intra_name)
    _closest_distances_intra = np.ndarray(closest_distances_intra_shape, dtype=DISTANCE_DTYPE, buffer=_closest_distances_intra_shm.buf)
    global _closest_points_intra_shm, _closest_points_intra
    _closest_points_intra_shm = shm.SharedMemory(name=closest_points_intra_name)
    _closest_points_intra = np.ndarray(closest_points_intra_shape, dtype=np.int32, buffer=_closest_points_intra_shm.buf)
    global _distances_inter_numerator_shm, _distances_inter_numerator
    _distances_inter_numerator_shm = shm.SharedMemory(name=distances_inter_numerator_name)
    _distances_inter_numerator = np.ndarray(distances_inter_numerator_shape, dtype=DISTANCE_DTYPE, buffer=_distances_inter_numerator_shm.buf)
    global _distances_inter_denominator_shm, _distances_inter_denominator
    _distances_inter_denominator_shm = shm.SharedMemory(name=distances_inter_denominator_name)
    _distances_inter_denominator = np.ndarray(distances_inter_denominator_shape, dtype=DISTANCE_DTYPE, buffer=_distances_inter_denominator_shm.buf)
    global _unique_clusters, _cost_per_cluster, _num_points, _num_clusters, _beta, _logsum_factor
    _unique_clusters = unique_clusters
    _cost_per_cluster = cost_per_cluster
    _num_points = num_points
    _num_clusters = num_clusters
    _beta = beta
    _logsum_factor = logsum_factor

    # Define clean up function to close shared memory
    def cleanup():
        try:
            _distances_shm.close()
            _clusters_shm.close()
            _closest_distances_intra_shm.close()
            _closest_points_intra_shm.close()
            _distances_inter_numerator_shm.close()
            _distances_inter_denominator_shm.close()
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
        A tuple containing the move type, indices to add, indices to remove, 
        candidate objective, add_within_cluster, and add_for_other_clusters if 
        an improvement is found, otherwise (None, None, None, -1, None, None).
    """
    global _distances, _clusters, _closest_distances_intra, _closest_points_intra, _closest_distances_inter, _closest_points_inter
    global _unique_clusters, _selection_cost, _num_points

    num_improvements = 0
    num_moves = 0
    for task, content in batch:
        if event.is_set():
            return None, None, None, -1, None, None
        if task == "add":
            idx_to_add = content
            cluster = _clusters[idx_to_add]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_add_mp(idx_to_add, objective, selection_per_cluster, nonselection_per_cluster[cluster])
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 0
                event.set()
                return "add", [idx_to_add], [], candidate_objective, add_within_cluster, add_for_other_clusters
        elif task == "swap" or task == "doubleswap":
            idxs_to_add, idx_to_remove = content
            cluster = _clusters[idx_to_remove]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_swap_mp(idxs_to_add, idx_to_remove, objective, selection_per_cluster, nonselection_per_cluster[cluster])
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 1
                event.set()
                return task, list(idxs_to_add), [idx_to_remove], candidate_objective, add_within_cluster, add_for_other_clusters
        elif task == "remove":
            idx_to_remove = content
            cluster = _clusters[idx_to_remove]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_remove_mp(idx_to_remove, objective, selection_per_cluster, nonselection_per_cluster[cluster])
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 1
                event.set()
                return "remove", [], [idx_to_remove], candidate_objective, add_within_cluster, add_for_other_clusters

    return None, None, None, -1, None, None

def process_batch_avg(batch, event, selection_per_cluster, nonselection_per_cluster, objective):
    """
    Processes a batch of tasks (used with multiprocessing).
    NOTE: this version is specifically intended for solution objects that
    use the "average" costs for inter-cluster distances!

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
        A tuple containing the move type, indices to add, indices to remove, 
        candidate objective, add_within_cluster, and add_for_other_clusters if 
        an improvement is found, otherwise (None, None, None, -1, None, None).
    """
    global _distances, _clusters, _closest_distances_intra, _closest_points_intra, _distances_inter_numerator, _distances_inter_denominator
    global _unique_clusters, _num_points, _num_clusters, _beta, _logsum_factor

    num_improvements = 0
    num_moves = 0
    for task, content in batch:
        if event.is_set():
            return None, None, None, -1, None, None
        if task == "add":
            idx_to_add = content
            cluster = _clusters[idx_to_add]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_add_mp_avg(idx_to_add, objective, selection_per_cluster, nonselection_per_cluster[cluster])
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 0
                event.set()
                return "add", [idx_to_add], [], candidate_objective, add_within_cluster, add_for_other_clusters
        elif task == "swap" or task == "doubleswap":
            idxs_to_add, idx_to_remove = content
            cluster = _clusters[idx_to_remove]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_swap_mp_avg(idxs_to_add, idx_to_remove, objective, selection_per_cluster, nonselection_per_cluster[cluster])
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 1
                event.set()
                return task, list(idxs_to_add), [idx_to_remove], candidate_objective, add_within_cluster, add_for_other_clusters
        elif task == "remove":
            idx_to_remove = content
            cluster = _clusters[idx_to_remove]
            candidate_objective, add_within_cluster, add_for_other_clusters = evaluate_remove_mp_avg(idx_to_remove, objective, selection_per_cluster, nonselection_per_cluster[cluster])
            num_moves += 1
            if candidate_objective > -1:
                num_improvements += 1
                event.set()
                return "remove", [], [idx_to_remove], candidate_objective, add_within_cluster, add_for_other_clusters

    return None, None, None, -1, None, None

def process_batch_result(result, results_list):
    """
    Adds the result of a move evaluation to the results list if the candidate objective is
    an improvement (otherwise it is ignored).
    NOTE: this modifies the results_list in place.

    Parameters:
    -----------
    result: tuple
        A tuple containing the move type, indices to add, indices to remove, 
        candidate objective, add_within_cluster, and add_for_other_clusters.
    results_list: list
        A list to which the result will be added if the candidate objective is an improvement.
        NOTE: This list should be managed by a Manager in a multiprocessing context.
    """
    move_type, idxs_to_add, idxs_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters = result
    if candidate_objective > -1:
        results_list.append((move_type, idxs_to_add, idxs_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters))

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