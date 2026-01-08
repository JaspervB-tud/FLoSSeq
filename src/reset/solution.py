# Set number of threads for various libraries to 1 to avoid oversubscription
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Import libraries
import numpy as np
from scipy.spatial.distance import squareform
import itertools
import math
import multiprocessing.shared_memory as shm
import multiprocessing as mp
from queue import Empty, Full
import time
import traceback

# This is to define the precision threshold for floating point comparisons
PRECISION_THRESHOLD = 1e-10
DISTANCE_DTYPE = np.float64
AUXILIARY_DISTANCE_DTYPE = np.float64

# This encodes move types for multiprocessing local search
MOVE_ADD = 0
MOVE_SWAP = 1
MOVE_DSWAP = 2
MOVE_REMOVE = 3
MOVE_BATCH = 4
MOVE_SYNC = 99
MOVE_STOP = 100

# Global variable for multiprocessing worker solutions (i.e. attached copies)
_WORKER_SOL = None

# Single processing solution class (stable)
class Solution:
    def __init__(self, distances: np.ndarray, clusters: np.ndarray, selection: np.ndarray = None, selection_cost: float = 1.0, cost_per_cluster: int = 0, scale: float = None, seed=None):
        """
        Initialize a Solution object (non-shared memory version).

        Parameters:
        -----------
        distances: numpy.ndarray or generator
            Either a 2D distance matrix, OR a generator that yields (i, j, distance) tuples.
            NOTE: The generator should yield all pairwise distances in condensed format (i.e. i and j are indices).
        clusters: numpy.ndarray
            A 1D array where clusters[i] represents the cluster assignment of point i.
        selection: numpy.ndarray, optional
            A 1D boolean array indicating which points are selected.
        selection_cost: float
            The cost associated with selecting a point.
        cost_per_cluster: int
            Defines how the cost of selecting a point in each cluster is calculated.
        scale: float, optional
            Scaling factor for inter-cluster distances in the objective function.
            NOTE: If None, no scaling is applied.
        seed: int or np.random.RandomState, optional
            Random seed for reproducibility.

        """
        
        import types
        # Check if distances is a numpy array or a generator
        is_generator = isinstance(distances, types.GeneratorType)
        is_array = isinstance(distances, np.ndarray)
        if not is_generator and not is_array:
            raise TypeError("distances must be a generator or a numpy array.")
        if is_array:
            # If distances is matrix, assert that distances and clusters have the same number of rows/points
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
        # If scale is provided, check if it is valid, otherwise use default (no scaling)
        if scale is not None:
            try:
                scale = float(scale)
                if scale < 0:
                    raise ValueError("Scale must be non-negative.")
                set_scale = True
            except:
                raise TypeError("Scale must be a float.")
        else:
            set_scale = False

        # Set random state for reproducibility
        if isinstance(seed, int):
            self.random_state = np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            self.random_state = seed
        else:
            self.random_state = np.random.RandomState()

        # Initialize basic attributes
        self.num_points = np.int64(clusters.shape[0])
        self.selection_cost = selection_cost

        # Process clusters
        unique_clusters, inv = np.unique(clusters, return_inverse=True)
        self.clusters = inv.astype(dtype=np.int64)
        self.unique_clusters = np.arange(unique_clusters.shape[0], dtype=np.int64)
        self.original_clusters = unique_clusters #store original cluster ids for reference
        self.num_clusters = unique_clusters.shape[0]
        # If scale is set, re-scale to same range as intra-costs and multiply by provided scale
        if set_scale:
            self.scale = scale * (self.num_points - self.num_clusters) / ((self.num_clusters * (self.num_clusters - 1)) / 2) #same scale as intra-cluster distances
        else: #if no scale, use default of 1.0 (no re-scaling)
            self.scale = 1.0

        # If distances is array, copy directly
        if is_array:
            flat_distances = squareform(distances, force='tovector', checks=False)
            self.distances = flat_distances.astype(dtype=DISTANCE_DTYPE)
        else: #otherwise stream distances into a flat array
            self.distances = np.zeros((self.num_points * (self.num_points - 1)) // 2, dtype=DISTANCE_DTYPE)
            for i, j, dist in distances:
                if not (0 <= dist <= 1):
                    raise ValueError(f"Distance at ({i}, {j}) = {dist} is not in range [0, 1].")
                min_idx = min(i, j)
                max_idx = max(i, j)
                idx = get_index(min_idx, max_idx, self.num_points)
                self.distances[idx] = dist

        # Initialize selection
        self.selection = selection.astype(dtype=bool)

        ################################ Initialize cost per cluster ################################
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
                centroid_idx = np.argmin([np.sum([get_distance(p, q, self.distances, self.num_points) for q in cluster_points]) for p in cluster_points])
                centroid = cluster_points[centroid_idx]
                self.cost_per_cluster[cluster] = np.mean([get_distance(centroid, p, self.distances, self.num_points) for p in cluster_points])
            elif cost_per_cluster == -2:
                # Define the average distance in a cluster as the average similarity
                # of all points in the cluster to the centroid of the cluster.
                cluster_points = np.where(self.clusters == cluster)[0]
                centroid_idx = np.argmin([np.sum([get_distance(p, q, self.distances, self.num_points) for q in cluster_points]) for p in cluster_points])
                centroid = cluster_points[centroid_idx]
                self.cost_per_cluster[cluster] = selection_cost * ( 1.0 - np.mean([get_distance(centroid, p, self.distances, self.num_points) for p in cluster_points]) )
            elif cost_per_cluster == 3:
                # Define the average distance in a cluster as the average distance
                # of all points in the cluster to the closest point in the cluster.
                cluster_points = np.where(self.clusters == cluster)[0]
                self.cost_per_cluster[cluster] = np.mean([np.min([get_distance(p, q, self.distances, self.num_points) for q in cluster_points if p != q]) for p in cluster_points])

        # Calculate objective
        self.calculate_objective()
        
    @classmethod
    def generate_centroid_solution(cls, distances, clusters, selection_cost: float = 1.0, cost_per_cluster: int = 0, scale: float = None, seed=None):
        """
        Generates a Solution object with an initial solution by selecting the centroid for every cluster.
        NOTE: This currently only works if distances is provided as a full distance matrix.

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

        return cls(distances, clusters, selection=selection, selection_cost=selection_cost, cost_per_cluster=cost_per_cluster, scale=scale, seed=seed)
    
    @classmethod
    def generate_random_solution(cls, distances, clusters, selection_cost: float = 1.0, cost_per_cluster: int = 0, scale: float = None, max_fraction=0.1, seed=None):
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
        remaining_points = np.flatnonzero(~selection)
        num_additional_points = max(0, max_selected_points - np.sum(selection))
        additional_points = random_state.choice(remaining_points, size=num_additional_points, replace=False)
        selection[additional_points] = True

        return cls(distances, clusters, selection, selection_cost=selection_cost, cost_per_cluster=cost_per_cluster, scale=scale, seed=random_state)

    # Core state and feasibility methods
    def determine_feasibility(self):
        """
        Determines if the solution stored in this object is feasible.
        NOTE: A solution is feasible if every cluster has at least one selected point.
        """
        uncovered_clusters = set(self.unique_clusters)
        for point in np.flatnonzero(self.selection):
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
        components = np.zeros(3, dtype=np.longdouble) #selection, intra, inter
        # Selection cost
        for idx in np.flatnonzero(self.selection):
            components[0] += self.cost_per_cluster[self.clusters[idx]]

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
                components[1] += cur_min

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
            components[2] += cur_max

        self.components = components
        self.objective = np.longdouble(components[0] + components[1] + (components[2] * self.scale)) #final objective rescaled
        return self.objective

    # Local search evaluation and acceptance methods
    def evaluate_add(self, idx_to_add: int):
        """
        Evaluates the effect of adding an unselected point to the solution.

        Parameters:
        -----------
        idx_to_add: int
            The index of the point to be added.
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the addition.
        candidate_components: np.ndarray
            The components of the objective after the addition.
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
        candidate_components = self.components.copy()
        candidate_components[0] += self.cost_per_cluster[cluster] #update selection component

        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for idx in self.nonselection_per_cluster[cluster]:
            cur_dist = get_distance(idx, idx_to_add, self.distances, self.num_points) #distance to current point (idx)
            if cur_dist < self.closest_distances_intra[idx]:
                candidate_components[1] += cur_dist - self.closest_distances_intra[idx] #update intra component
                add_within_cluster.append((idx, idx_to_add, cur_dist))

        # Calculate inter-cluster distances for other clusters (only if scale > 0)
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        if self.scale > 0:
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
                        candidate_components[2] += cur_max - self.closest_distances_inter[cluster, other_cluster] #update inter component
                        add_for_other_clusters.append((other_cluster, (idx_to_add, cur_idx), cur_max))

        candidate_objective = np.longdouble(candidate_components[0] + candidate_components[1] + candidate_components[2] * self.scale) #final objective rescaled
        return candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters

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
        candidate_components: np.ndarray
            The components of the objective after the swap.
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
        candidate_components = self.components.copy() 
        candidate_components[0] += (num_to_add - 1) * self.cost_per_cluster[cluster] #update selection component

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
                candidate_components[1] += cur_closest_distance - self.closest_distances_intra[idx] #update intra component
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
            else: #point to be removed is not closest, check if one of newly added points is closer
                cur_dists = [(get_distance(idx, idx_to_add, self.distances, self.num_points), idx_to_add) for idx_to_add in idxs_to_add]
                cur_dist, idx_to_add = min(cur_dists, key=lambda x: x[0])
                if cur_dist < cur_closest_distance:
                    candidate_components[1] += cur_dist - self.closest_distances_intra[idx] #update intra component
                    add_within_cluster.append((idx, idx_to_add, cur_dist))

        # Calculate inter-cluster distances for all other clusters (only if scale > 0)
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        if self.scale > 0:
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
                        candidate_components[2] += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster] #update inter component
                        add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))

        candidate_objective = np.longdouble(candidate_components[0] + candidate_components[1] + candidate_components[2] * self.scale) #final objective rescaled
        return candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters

    def evaluate_remove(self, idx_to_remove: int):
        """
        Evaluates the effect of removing a selected point from the solution.

        Parameters:
        -----------
        idx_to_remove: int
            The index of the point to be removed.
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the removal.
        candidate_components: np.ndarray
            The components of the objective after the removal.
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
        candidate_components = self.components.copy() 
        candidate_components[0] -= self.cost_per_cluster[cluster] #update selection component
        
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
                candidate_components[1] += cur_closest_distance - self.closest_distances_intra[idx] #update intra component
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))

        # Calculate inter-cluster distances for all other clusters (only if scale > 0)
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        if self.scale > 0:
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
                        candidate_components[2] += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster] #update inter component
                        add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
        
        candidate_objective = np.longdouble(candidate_components[0] + candidate_components[1] + candidate_components[2] * self.scale) #final objective rescaled
        return candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters

    def accept_move(self, idxs_to_add: list, idxs_to_remove: list, candidate_objective: float, candidate_components: np.ndarray, add_within_cluster: list, add_for_other_clusters: list):
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
        candidate_components: np.ndarray
            The components of the objective after the move.
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

        self.components = candidate_components
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
        selected = sorted(list(self.selection_per_cluster[cluster])) #sorted to ensure consistent ordering
        unselected = sorted(list(self.nonselection_per_cluster[cluster]))

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
        indices = np.flatnonzero(self.selection)
        if random:
            for idx in self.random_state.permutation(indices):
                if len(self.selection_per_cluster[self.clusters[idx]]) > 1:
                    yield idx
        else:
            for idx in indices:
                if len(self.selection_per_cluster[self.clusters[idx]]) > 1:
                    yield idx

    # Local search (single processing, for multiprocessing see Solution_shm)
    def local_search(self,
                    max_iterations: int = np.inf, max_runtime: float = np.inf,
                    random_move_order: bool = True, random_index_order: bool = True, move_order: list = ["add", "swap", "doubleswap", "remove"],
                    doubleswap_time_threshold: float = 60.0,
                    logging: bool = False, logging_frequency: int = 100,
                    ):
        """
        Perform local search to find a (local) optimal solution using a single processor. 
        
        Parameters:
        -----------
        max_iterations: int
            The maximum number of iterations to perform. Default is infinity.
        max_runtime: float
            The maximum runtime in seconds for the local search. Default is infinity.
        random_move_order: bool
            If True, the order of moves is randomized. Default is True.
        random_index_order: bool
            If True, the order of indices for moves is randomized. Default is True.
            NOTE: if random_move_order is True, but this is false,
            all moves of a particular type will be exhausted before moving to the next type,
            but the order of moves is random.
        move_order: list
            If provided, this list will be used to determine the
            order of moves. If random_move_order is True, this
            list will be shuffled before use.
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
        logging: bool
            If True, information about the local search will be printed. Default is False.
        logging_frequency: int
            If logging is True, information will be printed every logging_frequency iterations. Default is 100.

        Returns:
        --------
        time_per_iteration: list of floats
            The time taken for each iteration.
            NOTE: this is primarily for logging purposes
        objectives: list of floats
            The objective value after each iteration.
        """
        # Validate input
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

        print(f"Starting local search with objective {self.objective:.6f}", flush=True)
        start_time = time.time()
        while iteration < max_iterations:
            if time.time() - start_time > max_runtime:
                print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time:.2f} seconds). Stopping local search.", flush=True)
                break

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

            # Helper function for getting next task and handling doubleswap time threshold
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
                        
                    # Select next move type and content
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
            
            move_counter = 0
            current_iteration_time = time.time()
            while active_moves and not solution_changed:
                # Select next move type
                move_code, move_content = next_task()
                if move_code is None:
                    break #no more moves available

                # Evaluate move
                if move_code == MOVE_ADD:
                    idx_to_add = move_content
                    idxs_to_add = [idx_to_add]
                    idxs_to_remove = []
                    candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                        solution_changed = True
                        break
                elif move_code == MOVE_SWAP or move_code == MOVE_DSWAP:
                    idxs_to_add, idx_to_remove = move_content
                    idxs_to_remove = [idx_to_remove]
                    candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idxs_to_add, idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                        solution_changed = True
                        break
                elif move_code == MOVE_REMOVE:
                    idx_to_remove = move_content
                    idxs_to_add = []
                    idxs_to_remove = [idx_to_remove]
                    candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove)
                    if candidate_objective < self.objective and np.abs(candidate_objective - self.objective) > PRECISION_THRESHOLD:
                        solution_changed = True
                        break
                move_counter += 1

                # Check runtime
                if move_counter % 500 == 0:
                    # Check if total runtime exceeds max_runtime
                    if time.time() - start_time > max_runtime:
                        if logging:
                            print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time} seconds). Stopping local search.", flush=True)
                        return time_per_iteration, objectives
                        

            if solution_changed: # If improvement is found, update solution
                self.accept_move(idxs_to_add, idxs_to_remove, candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters)
                del idxs_to_add, idxs_to_remove, candidate_objective, add_within_cluster, add_for_other_clusters #sanity check, should throw error if something weird happens
                
                # Update time tracking and iteration counter
                time_per_iteration.append(time.time() - current_iteration_time)
                iteration += 1

                # Check if time exceeds allowed runtime
                if time.time() - start_time > max_runtime:
                    if logging:
                        print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time} seconds). Stopping local search.", flush=True)
                    return time_per_iteration, objectives
            
                if logging and (iteration % logging_frequency == 0):
                    print(f"Iteration {iteration}: Objective = {self.objective:.10f} - selection_cost={self.components[0]:.10f}, intra-cost={self.components[1]:.10f}, inter-cost={self.components[2]:.10f}", flush=True)
                    print(f"Average runtime last {logging_frequency} iterations: {np.mean(time_per_iteration[-logging_frequency:]):.6f} seconds", flush=True)
            else: #solution did not change -> local optimum found!
                break

        return time_per_iteration, objectives

    # Equality check
    def __eq__(self, other):
        """
        Check if two solutions are equal.
        NOTE: This purely checks if all relevant attributes are equal, excluding the random state.
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
        # Check if scales are equal
        if not math.isclose(self.scale, other.scale, rel_tol=PRECISION_THRESHOLD):
            print("Scales are not equal.")
            return False
        # Check if objectives are equal
        if not math.isclose(self.objective, other.objective, rel_tol=PRECISION_THRESHOLD):
            print("Objectives are not equal.")
            return False
        # Check if components are equal
        try:
            if not np.allclose(self.components, other.components, atol=PRECISION_THRESHOLD):
                print("Components are not equal.")
                return False
        except:
            print("Components could not be compared.")
            return False
        return True

# Multiprocessing Solution class using shared memory (stable)
class Solution_shm(Solution):
    def __init__(self, distances, clusters: np.ndarray, selection: np.ndarray = None, selection_cost: float = 1.0, cost_per_cluster: int = 0, scale: float = None, shm_prefix: str = None, seed=None):
        """
        Initialize a Solution object using shared memory arrays.
        
        Parameters:
        -----------
        distances: numpy.ndarray or generator
            Either a 2D distance matrix, OR a generator that yields (i, j, distance) tuples.
            NOTE: The generator should yield all pairwise distances in condensed format (i.e. i and j are indices).
        clusters: numpy.ndarray
            A 1D array where clusters[i] represents the cluster assignment of point i.
        selection: numpy.ndarray, optional
            A 1D boolean array indicating which points are selected.
        selection_cost: float
            The cost associated with selecting a point.
        cost_per_cluster: int
            Defines how the cost of selecting a point in each cluster is calculated.
        scale: float, optional
            Scaling factor for inter-cluster distances in the objective function.
            NOTE: If None, no scaling is applied.
        shm_prefix: str, optional
            Prefix for shared memory segment names. If None, a unique prefix is generated.
        seed: int or np.random.RandomState, optional
            Random seed for reproducibility.
        """
        import types
        # Check if distances is a generator, array, or other
        is_generator = isinstance(distances, types.GeneratorType)
        is_array = isinstance(distances, np.ndarray)
        if not is_generator and not is_array:
            raise TypeError("distances must be a generator or a numpy array.")
        if is_array:
            # If distances is matrix, assert that distances and clusters have the same number of rows/points
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
        # If scale is provided, check if it is valid, otherwise use default (no scaling)
        if scale is not None:
            try:
                scale = float(scale)
                if scale < 0:
                    raise ValueError("Scale must be non-negative.")
                set_scale = True
            except:
                raise TypeError("Scale must be a float.")
        else:
            set_scale = False

        # Set random state for reproducibility
        if isinstance(seed, int):
            self.random_state = np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            self.random_state = seed
        else:
            self.random_state = np.random.RandomState()

        # Initialize basic attributes
        self.num_points = np.int64(clusters.shape[0])
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
        # If scale is set, re-scale to same range as intra-costs and multiply by provided scale
        self._create_shm_array("scale", (1,), np.float64)
        if set_scale:
            self.scale[0] = scale * (self.num_points - self.num_clusters) / ((self.num_clusters * (self.num_clusters - 1)) / 2)
        else:
            self.scale[0] = 1.0

        # Calculate condensed distance matrix size
        condensed_size = (self.num_points * (self.num_points - 1)) // 2
        
        # Create shared memory for distances and stream data directly
        self._create_shm_array("distances", (condensed_size,), DISTANCE_DTYPE)

        # If distances is array, copy directly
        if is_array:
            flat_distances = squareform(distances, force="tovector", checks=False)
            np.copyto(self.distances, flat_distances.astype(dtype=DISTANCE_DTYPE))
        else: #otherwise stream distances into shared memory
            for i, j, dist in distances:
                if not (0 <= dist <= 1):
                    raise ValueError(f"Distance at ({i}, {j}) = {dist} is not in range [0, 1].")
                min_idx = min(i, j)
                max_idx = max(i, j)
                idx = get_index(min_idx, max_idx, self.num_points)
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
                # Define the average distance in a cluster as the average distance
                # of all points in the cluster to the closest point in the cluster.
                cluster_points = np.where(self.clusters == cluster)[0]
                self.cost_per_cluster[cluster] = np.mean([np.min([get_distance(p, q, self.distances, self.num_points) for q in cluster_points if p != q]) for p in cluster_points])      
        
        # Build CSR representation for clusters
        self._create_shm_array("cluster_members", (self.num_points,), np.int64)
        self._create_shm_array("cluster_offsets", (self.unique_clusters.shape[0]+1,), np.int64)

        order = np.argsort(self.clusters)
        np.copyto( self.cluster_members, order.astype(np.int64, copy=False))

        counts = np.bincount(self.clusters, minlength=self.unique_clusters.shape[0]).astype(np.int64)
        self.cluster_offsets[1:] = np.cumsum(counts)

        # Calculate objective
        self._create_shm_array("objective", (1,), np.longdouble)
        self._create_shm_array("components", (3,), np.longdouble) #selection, intra, inter
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
        _attach("scale", (1,), np.float64)
        _attach("objective", (1,), np.longdouble)
        _attach("components", (3,), np.longdouble)

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

    def __del__(self):
        """
        Close shared memory resources without unlinking.
        Used to ensure that shared memory is not unlinked prematurely.
        """
        for handle in self._shm_handles.values():
            try:
                handle.close()
            except Exception as e:
                pass # Silent failure during close

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
            self.objective[0] = np.longdouble(np.inf)
            return self.objective
        self.feasible = True

        # Calculate objective value
        objective = 0.0
        components = np.zeros(3, dtype=np.longdouble) #selection, intra, inter

        self.num_selected_per_cluster.fill(0)
        # Selection cost
        for idx in np.flatnonzero(self.selection):
            components[0] += self.cost_per_cluster[self.clusters[idx]]
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
                components[1] += cur_min

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
            components[2] += cur_max

        self.components[:] = components
        self.objective[0] = np.longdouble(components[0] + components[1] + components[2] * self.scale[0]) #final objective rescaled
        return self.objective[0]

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
    def evaluate_add(self, idx_to_add: int, stop_event = None):
        """
        Evaluates the effect of adding an unselected point to the solution.

        Parameters:
        -----------
        idx_to_add: int
            The index of the point to be added.
        stop_event: multiprocessing.Event, optional
            An optional event that can be used to signal early termination of the evaluation.
            If the event is set during evaluation, the method will return (np.inf, None, None).
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the addition.
        candidate_components: np.ndarray
            The components of the objective after the addition.
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
            return np.inf, None, None, None

        # Find current cluster
        cluster = self.clusters[idx_to_add]

        # Calculate selection cost
        candidate_components = self.components.copy()
        candidate_components[0] += self.cost_per_cluster[cluster]

        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for i, idx in enumerate(self.iter_unselected(cluster)):
            if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                return np.inf, None, None, None
            cur_dist = get_distance(idx, idx_to_add, self.distances, self.num_points) #distance to current point (idx)
            if cur_dist < self.closest_distances_intra[idx]:
                candidate_components[1] += cur_dist - self.closest_distances_intra[idx] #update intra component
                add_within_cluster.append((idx, idx_to_add, cur_dist))

        # Calculate inter-cluster distances for other clusters (only if scale > 0)
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        if self.scale[0] > 0:
            for other_cluster in self.unique_clusters:
                if stop_event is not None and stop_event.is_set():
                    return np.inf, None, None, None
                if other_cluster != cluster:
                    cur_max = self.closest_distances_inter[cluster, other_cluster]
                    cur_idx = -1
                    for i, idx in enumerate(self.iter_selected(other_cluster)):
                        if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                            return np.inf, None, None, None
                        cur_similarity = 1 - get_distance(idx, idx_to_add, self.distances, self.num_points) #this is the similarity, if it is more similar then change solution
                        if cur_similarity > cur_max:
                            cur_max = cur_similarity
                            cur_idx = idx
                    if cur_idx > -1:
                        candidate_components[2] += cur_max - self.closest_distances_inter[cluster, other_cluster] #update inter component
                        add_for_other_clusters.append((other_cluster, (idx_to_add, cur_idx), cur_max))

        candidate_objective = np.longdouble(candidate_components[0] + candidate_components[1] + candidate_components[2] * self.scale[0]) #final objective rescaled
        return candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters

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
        candidate_components: np.ndarray
            The components of the objective after the swap.
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
            return np.inf, None, None, None
            
        # Generate pool of alternative points to compare to
        new_selection = set(self.iter_selected(cluster))
        for idx in idxs_to_add:
            new_selection.add(idx)
        new_selection.remove(idx_to_remove)
        new_nonselection = set(self.iter_unselected(cluster))
        new_nonselection.add(idx_to_remove)

        # Calculate selection cost
        candidate_components = self.components.copy()
        candidate_components[0] += (num_to_add - 1) * self.cost_per_cluster[cluster] #update selection component

        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for i, idx in enumerate(new_nonselection):
            if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                return np.inf, None, None, None
            cur_closest_distance = self.closest_distances_intra[idx]
            cur_closest_point = self.closest_points_intra[idx]
            if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                cur_closest_distance = np.inf
                for other_idx in new_selection:
                    cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                    if cur_dist < cur_closest_distance:
                        cur_closest_distance = cur_dist
                        cur_closest_point = other_idx
                candidate_components[1] += cur_closest_distance - self.closest_distances_intra[idx] #update intra component
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
            else: #point to be removed is not closest, check if one of newly added points is closer
                cur_dists = [(get_distance(idx, idx_to_add, self.distances, self.num_points), idx_to_add) for idx_to_add in idxs_to_add]
                cur_dist, idx_to_add = min(cur_dists, key=lambda x: x[0])
                if cur_dist < cur_closest_distance:
                    candidate_components[1] += cur_dist - self.closest_distances_intra[idx] #update intra component
                    add_within_cluster.append((idx, idx_to_add, cur_dist))

        # Calculate inter-cluster distances for all other clusters (only if scale > 0)
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        if self.scale[0] > 0:
            for other_cluster in self.unique_clusters:
                if stop_event is not None and stop_event.is_set():
                    return np.inf, None, None, None
                if other_cluster != cluster:
                    cur_closest_similarity = self.closest_distances_inter[cluster, other_cluster]
                    cur_closest_point = self.closest_points_inter[cluster, other_cluster]
                    cur_closest_pair = (-1, -1)
                    if cur_closest_point == idx_to_remove: #if point to be removed is closest for current, find new closest
                        cur_closest_similarity = -np.inf
                        for i, idx in enumerate(self.iter_selected(other_cluster)):
                            if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                                return np.inf, None, None, None
                            for other_idx in new_selection:
                                cur_similarity = 1.0 - get_distance(idx, other_idx, self.distances, self.num_points)
                                if cur_similarity > cur_closest_similarity:
                                    cur_closest_similarity = cur_similarity
                                    cur_closest_pair = (other_idx, idx)
                    else: #point to be removed is not closest, check if one of newly added points is closer
                        for i, idx in enumerate(self.iter_selected(other_cluster)):
                            if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                                return np.inf, None, None, None
                            cur_similarities = [(1.0 - get_distance(idx, idx_to_add, self.distances, self.num_points), idx_to_add) for idx_to_add in idxs_to_add]
                            cur_similarity, idx_to_add = max(cur_similarities, key = lambda x: x[0])
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                cur_closest_pair = (idx_to_add, idx)
                    if cur_closest_pair[0] > -1:
                        candidate_components[2] += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster] #update inter component
                        add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))

        candidate_objective = np.longdouble(candidate_components[0] + candidate_components[1] + candidate_components[2] * self.scale[0]) #final objective rescaled
        return candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters

    def evaluate_remove(self, idx_to_remove: int, stop_event = None):
        """
        Evaluates the effect of removing a selected point from the solution.

        Parameters:
        -----------
        idx_to_remove: int
            The index of the point to be removed.
        stop_event: multiprocessing.Event, optional
            An optional event that can be used to signal early termination of the evaluation.
            If the event is set during evaluation, the method will return (np.inf, None, None).
        
        Returns:
        --------
        candidate_objective: float
            The objective value of the solution after the removal.
        candidate_components: np.ndarray
            The components of the objective after the removal.
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
            return np.inf, None, None, None
        
        # Find current cluster
        cluster = self.clusters[idx_to_remove]

        # Generate pool of alternative points to compare to
        new_selection = set(self.iter_selected(cluster))
        new_selection.remove(idx_to_remove)
        new_nonselection = set(self.iter_unselected(cluster))
        new_nonselection.add(idx_to_remove)

        # Calculate selection cost
        candidate_components = self.components.copy()
        candidate_components[0] -= self.cost_per_cluster[cluster] #update selection component

        # Calculate intra-cluster distances
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for i, idx in enumerate(new_nonselection):
            if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                return np.inf, None, None, None
            cur_closest_point = self.closest_points_intra[idx]
            if cur_closest_point == idx_to_remove:
                cur_closest_distance = np.inf
                for j, other_idx in enumerate(new_selection):
                    if stop_event is not None and (j & 63)==0 and stop_event.is_set(): #check every 64 iterations
                        return np.inf, None, None, None
                    if other_idx != idx:
                        cur_dist = get_distance(idx, other_idx, self.distances, self.num_points)
                        if cur_dist < cur_closest_distance:
                            cur_closest_distance = cur_dist
                            cur_closest_point = other_idx
                candidate_components[1] += cur_closest_distance - self.closest_distances_intra[idx] #update intra component
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))

        # Calculate inter-cluster distances for all other clusters (only if scale > 0)
        add_for_other_clusters = [] #this stores changes that have to be made if the objective improves
        if self.scale[0] > 0:
            for other_cluster in self.unique_clusters:
                if other_cluster != cluster:
                    cur_closest_similarity = self.closest_distances_inter[cluster, other_cluster]
                    cur_closest_point = self.closest_points_inter[cluster, other_cluster]
                    cur_closest_pair = (-1, -1)
                    if cur_closest_point == idx_to_remove:
                        cur_closest_similarity = -np.inf
                        for i, idx in enumerate(self.iter_selected(other_cluster)):
                            if stop_event is not None and (i & 63)==0 and stop_event.is_set(): #check every 64 iterations
                                return np.inf, None, None, None
                            for other_idx in new_selection:
                                cur_similarity = 1.0 - get_distance(idx, other_idx, self.distances, self.num_points)
                                if cur_similarity > cur_closest_similarity:
                                    cur_closest_similarity = cur_similarity
                                    cur_closest_pair = (other_idx, idx)
                        candidate_components[2] += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster] #update inter component
                        add_for_other_clusters.append((other_cluster, cur_closest_pair, cur_closest_similarity))
        
        candidate_objective = np.longdouble(candidate_components[0] + candidate_components[1] + candidate_components[2] * self.scale[0]) #final objective rescaled
        return candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters

    def accept_move(self, idxs_to_add: list, idxs_to_remove: list, candidate_objective: float, candidate_components: np.ndarray, add_within_cluster: list, add_for_other_clusters: list):
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
        candidate_components: np.ndarray
            The components of the objective after the move.
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

        self.components[:] = candidate_components
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
                    max_iterations: int = np.inf, max_runtime: float = np.inf,
                    random_move_order: bool = True, random_index_order: bool = True, move_order: list = ["add", "swap", "doubleswap", "remove"],
                    doubleswap_time_threshold: float = 60.0,
                    task_queue_size: int = 2000,
                    mp_switch_threshold: float = 10.0,
                    logging: bool = False, logging_frequency: int = 100,
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
            The maximum number of iterations to perform. Default is infinity.
        max_runtime: float
            The maximum runtime in seconds for the local search. Default is infinity.
        random_move_order: bool
            If True, the order of moves is randomized. Default is True.
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
        mp_switch_threshold: float
            The time threshold in seconds after which multiprocessing will be used. Default is 5.0 seconds.
            NOTE: if the local search iteration time is below this threshold, the local search will
            be performed in the main process only (without using worker processes), otherwise it will switch
            to multiprocessing for the current iteration.
        logging: bool
            If True, information about the local search will be printed. Default is False.
        logging_frequency: int
            If logging is True, information will be printed every logging_frequency iterations. Default is 100.

        Returns:
        --------
        time_per_iteration: list of floats
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
        if not isinstance(mp_switch_threshold, (int, float)) or mp_switch_threshold < 0:
            raise ValueError("mp_switch_threshold must be a non-negative number.")
        if not isinstance(logging, bool):
            raise ValueError("logging must be a boolean value.")
        if not isinstance(logging_frequency, int) or logging_frequency < 1:
            raise ValueError("logging_frequency must be a positive integer.")  
        if not self.feasible:
            raise ValueError("The solution is infeasible, cannot perform local search.")

        # Create epoch tag
        self.epoch[0] = 0

        # Start worker processes if num_processes > 1
        workers = None
        task_q = None
        result_q = None
        stop_event = None

        if num_processes > 1:
            # Create context variables and main process variables
            try:
                context = mp.get_context("spawn")
            except RuntimeError:
                context = mp.get_context() #fallback to default if spawn not available
            stop_event = context.Event()
            task_q = context.Queue(maxsize=task_queue_size)
            result_q = context.Queue()

            # Start worker processes
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
                        stop_event,
                    ),
                    daemon = True,
                )
                worker.start()
                workers.append(worker)

        def drain_queue(q):
            """Helper function to drain a queue."""
            if q is None:
                return
            try:
                while True:
                    q.get_nowait()
            except Empty:
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

                objectives.append(self.objective[0])
                solution_changed = False
                using_mp = False

                # If using multiprocessing, clear stop event and drain queues
                if stop_event is not None:
                    stop_event.clear()
                if result_q is not None:
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

                # Helper function for getting next task and handling doubleswap time threshold
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

                        # Select next move type and content
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

                move_counter = 0
                current_iteration_time = time.time()
                # Phase 1: single processing
                while active_moves and not solution_changed:
                    if (workers is not None and time.time() - current_iteration_time) > mp_switch_threshold:
                        if logging:
                            print(f"Iteration {iteration}: Switching to multiprocessing after {time.time() - current_iteration_time:.2f}s in single processing mode.", flush=True)
                        using_mp = True
                        break #switch to multiprocessing

                    # Select next move type
                    move_code, move_content = next_task()
                    if move_code is None:
                        break #no more moves available

                    # Evaluate move in single processing
                    if move_code  == MOVE_ADD:
                        idx_to_add = move_content
                        idxs_to_add = [idx_to_add]
                        idxs_to_remove = []
                        candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add)
                        if candidate_objective < self.objective[0] and abs(candidate_objective - self.objective[0]) > PRECISION_THRESHOLD:
                            solution_changed = True
                            break
                    elif move_code == MOVE_SWAP or move_code == MOVE_DSWAP:
                        idxs_to_add, idx_to_remove = move_content
                        idxs_to_remove = [idx_to_remove]
                        candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idxs_to_add, idx_to_remove)
                        if candidate_objective < self.objective[0] and abs(candidate_objective - self.objective[0]) > PRECISION_THRESHOLD:
                            solution_changed = True
                            break
                    elif move_code == MOVE_REMOVE:
                        idx_to_remove = move_content
                        idxs_to_add = []
                        idxs_to_remove = [idx_to_remove]
                        candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove)
                        if candidate_objective < self.objective[0] and abs(candidate_objective - self.objective[0]) > PRECISION_THRESHOLD:
                            solution_changed = True
                            break
                    move_counter += 1

                    # Check runtime
                    if move_counter % 500 == 0:
                        # Check if total runtime exceeds max_runtime
                        if time.time() - start_time > max_runtime:
                            if logging:
                                print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time:.2f} seconds). Stopping local search.", flush=True)
                            return time_per_iteration, objectives


                # Phase 2: multiprocessing
                if using_mp and not solution_changed:
                    epoch = int(self.epoch[0])
                    BATCH_SIZE = 64 #number of tasks to send in one batch
                    PREFETCH_BATCHES = 4 * num_processes

                    inflight = 0
                    all_tasks_distributed = False

                    # Process move evaluations using multiprocessing
                    while True:
                        # Prefill task queue
                        while (not all_tasks_distributed) and (inflight < PREFETCH_BATCHES):
                            batch = []
                            for _ in range(BATCH_SIZE):
                                move_code, move_content = next_task()
                                if move_code is None:
                                    all_tasks_distributed = True
                                    break
                                batch.append( (move_code, move_content) )

                            if not batch:
                                break #no more tasks to distribute

                            try:
                                task_q.put( (epoch, MOVE_BATCH, batch), timeout=0.001 )
                                inflight += 1
                            except Full:
                                if any(not w.is_alive() for w in workers):
                                    raise RuntimeError("Worker has died unexpectedly during local search.")
                                continue

                        # Break if all tasks have been distributed
                        if inflight == 0 and all_tasks_distributed:
                            break

                        # Wait for worker results
                        if time.time() - start_time > max_runtime:
                            return time_per_iteration, objectives
                        try:
                            cur_epoch, cur_move_type, cur_move_data = result_q.get(timeout=0.05)
                        except Empty:
                            continue

                        # Check epoch
                        if cur_epoch != epoch:
                            continue #stale result from previous epoch

                        if cur_move_type is None: #batch is done -> decrement inflight
                            inflight -= 1
                            continue

                        # Found improving move
                        solution_changed = True
                        break

                    if solution_changed:
                        stop_event.set() #notify workers to stop current evaluations

                        # Re-evaluate move to get updates
                        if cur_move_type == MOVE_ADD:
                            idx_to_add = cur_move_data
                            idxs_to_add = [idx_to_add]
                            idxs_to_remove = []
                            candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters = self.evaluate_add(idx_to_add)
                        elif cur_move_type == MOVE_SWAP or cur_move_type == MOVE_DSWAP:
                            idxs_to_add, idx_to_remove = cur_move_data
                            idxs_to_remove = [idx_to_remove]
                            candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters = self.evaluate_swap(idxs_to_add, idx_to_remove)
                        elif cur_move_type == MOVE_REMOVE:
                            idx_to_remove = cur_move_data
                            idxs_to_add = []
                            idxs_to_remove = [idx_to_remove]
                            candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters = self.evaluate_remove(idx_to_remove)

                # If solution changed (regardless of single or multiprocessing), accept the move
                if solution_changed:
                    self.accept_move(idxs_to_add, idxs_to_remove, candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters)
                    del idxs_to_add, idxs_to_remove, candidate_objective, candidate_components, add_within_cluster, add_for_other_clusters #sanity check, should throw error if something weird happens

                    # Update time tracking and epoch/iteration counters
                    time_per_iteration.append(time.time() - current_iteration_time)
                    iteration += 1
                    self.epoch[0] += 1

                    # Check if time exceeds allowed runtime
                    if time.time() - start_time > max_runtime:
                        if logging:
                            print(f"Max runtime of {max_runtime} seconds exceeded ({time.time() - start_time:.2f} seconds). Stopping local search.", flush=True)
                        return time_per_iteration, objectives

                    if logging and (iteration % logging_frequency == 0):
                        print(f"Iteration {iteration}: Objective = {self.objective[0]:.10f} - selection_cost={self.components[0]:.10f}, intra-cost={self.components[1]:.10f}, inter-cost={self.components[2]:.10f}", flush=True)
                        print(f"Average runtime last {logging_frequency} iterations: {np.mean(time_per_iteration[-logging_frequency:]):.6f} seconds", flush=True)
                else: #solution did not change -> local optimum found!
                    break

            return time_per_iteration, objectives
        
        finally:
            # Terminate workers
            if stop_event is not None:
                stop_event.set()
            if task_q is not None:
                drain_queue(task_q)
            if workers is not None and task_q is not None:
                for _ in workers:
                    try:
                        task_q.put( (int(self.epoch[0]), MOVE_STOP, None) )
                    except Full:
                        pass
                # Join workers
                for worker in workers:
                    worker.join(timeout=2.0)

    # Equality check
    def __eq__(self, other):
        """
        Check if two solutions are equal.
        NOTE: This purely checks if all relevant attributes are equal, excluding the random state and other shm-based properties.
        """
        # Check if other is an instance of the same class
        if not isinstance(other, Solution_shm):
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
        # Check if scales are equal
        if not math.isclose(self.scale[0], other.scale[0], rel_tol=PRECISION_THRESHOLD):
            print("Scales are not equal.")
            return False
        # Check if objectives are equal
        if not math.isclose(self.objective[0], other.objective[0], rel_tol=PRECISION_THRESHOLD):
            print("Objectives are not equal.")
            return False
        # Check if components are equal
        try:
            if not np.allclose(self.components, other.components, atol=PRECISION_THRESHOLD):
                print("Components are not equal.")
                return False
        except:
            print("Components could not be compared.")
            return False
        return True
        
# Module-level functions
def _shm_worker_main(shm_prefix, num_points, num_clusters, task_q, result_q, stop_event):
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

            # Terminate if stop event is set
            if stop_event.is_set():
                if move_code == MOVE_BATCH:
                    result_q.put( (epoch, None, None)) #keep track of inflight tasks
                continue

            sol = _WORKER_SOL
            cur_epoch = sol.epoch[0] if sol.epoch is not None else -1 #cache for consistency
            if cur_epoch != epoch:
                # Solution has changed, skip current evaluation
                continue

            # Process move or batch of moves
            if move_code == MOVE_BATCH:
                cur_obj = sol.objective[0]
                improved = False
                for mc, margs in move_args:
                    if stop_event.is_set():
                        break

                    cur_epoch = sol.epoch[0] if sol.epoch is not None else -1 #cache for consistency
                    if (cur_epoch != epoch):
                        break

                    if mc == MOVE_ADD:
                        idx_to_add = margs
                        candidate_objective, _, _, _ = sol.evaluate_add(idx_to_add, stop_event=stop_event)
                        if candidate_objective < cur_obj and abs(candidate_objective - cur_obj) > PRECISION_THRESHOLD:
                            # Suppress potentially late publishing of moves from old epochs
                            final_epoch = sol.epoch[0] if sol.epoch is not None else -1
                            if (not stop_event.is_set()) and (epoch == final_epoch):
                                result_q.put( (epoch, mc, idx_to_add) )
                                improved = True
                    elif mc == MOVE_SWAP or mc == MOVE_DSWAP:
                        idxs_to_add, idx_to_remove = margs
                        candidate_objective, _, _, _ = sol.evaluate_swap(idxs_to_add, idx_to_remove, stop_event=stop_event)
                        if candidate_objective < cur_obj and abs(candidate_objective - cur_obj) > PRECISION_THRESHOLD:
                            # Suppress potentially late publishing of moves from old epochs
                            final_epoch = sol.epoch[0] if sol.epoch is not None else -1
                            if (not stop_event.is_set()) and (epoch == final_epoch):
                                result_q.put( (epoch, mc, (idxs_to_add, idx_to_remove)) )
                                improved = True
                    elif mc == MOVE_REMOVE:
                        idx_to_remove = margs
                        candidate_objective, _, _, _ = sol.evaluate_remove(idx_to_remove, stop_event=stop_event)
                        if candidate_objective < cur_obj and abs(candidate_objective - cur_obj) > PRECISION_THRESHOLD:
                            # Suppress potentially late publishing of moves from old epochs
                            final_epoch = sol.epoch[0] if sol.epoch is not None else -1
                            if (not stop_event.is_set()) and (epoch == final_epoch):
                                result_q.put( (epoch, mc, idx_to_remove) )
                                improved = True

                    if improved: #if improving move is found, stop processing remaining moves in batch
                        break

                if not improved:
                    result_q.put( (epoch, None, None)) #keep track of inflight tasks
                continue
                
    finally:
        del _WORKER_SOL

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


############### MAIN FUNCTION FOR MANUSCRIPT ###############
def read_metadata(path):
    seq2lin = {}
    id_col = 0
    lineage_col = 13
    with open(path, "r") as f_in:
        # Don't skip header, header isn't included
        for line in f_in:
            parts = line.strip().split("\t")
            seq2lin[parts[id_col]] = parts[lineage_col]
    return seq2lin

def read_sequence_mapping(path):
    sequence_mapping = {}
    with open(path, "r") as f_in:
        for line in f_in:
            parts = line.strip().split("\t")
            idx = int(parts[0])
            seq_id = parts[1]
            sequence_mapping[idx] = seq_id
    return sequence_mapping

def generate_distances_mash(path):
    mash_indices = []
    try:
        with open(path, "r") as f_in: #Mash outputs distance pairs in tab-delimited format
            next(f_in) #skip header
            for line in f_in:
                parts = line.strip().split("\t")

                cur_idx = int(parts[0])
                for other_idx, d in enumerate(parts[1:]): #remainder of line are distances to previous indices
                    other_idx = mash_indices[other_idx]
                    d = float(d)

                    yield(cur_idx, other_idx, d)
                mash_indices.append(cur_idx)
    except FileNotFoundError:
        print("Mash output file not found.", flush=True)
        return #empty generator
                    
def generate_distances_sourmash(path, jaccard):
    QUERY_IDX = 0
    MATCH_IDX = 2
    JACCARD_IDX = 6
    COSINE_IDX = 12

    dist_idx = JACCARD_IDX if jaccard else COSINE_IDX
    try:
        with open(path, "r") as f_in: #Sourmash outputs distance pairs in csv format
            next(f_in) #skip header
            for line in f_in:
                parts = line.strip().split(",")

                # Parse line
                query = int(parts[QUERY_IDX])
                match = int(parts[MATCH_IDX])
                
                d = 1.0 - float(parts[dist_idx]) #convert similarity to distance

                yield (query, match, d)
    except FileNotFoundError:
        print("Sourmash output file not found.", flush=True)
        return #empty generator

def main():
    import argparse
    import time
    import numpy as np
    import math
    import traceback

    parser = argparse.ArgumentParser(description="Run local search on SARS-CoV-2 data.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata file.")
    parser.add_argument("--sequences_mapping", type=str, required=True, help="Path to sequences mapping file.")
    parser.add_argument("--distances", type=str, required=True, help="Path to distances file.")
    parser.add_argument("--mash", action="store_true", help="Use Mash distances if set, otherwise use sourmash distances.")
    parser.add_argument("--jaccard", action="store_true", help="Use Jaccard distances for sourmash if set, otherwise use cosine distances.")
    parser.add_argument("--selection_cost", type=float, default=1.0, help="Selection cost per selected point.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for solution initialization.")
    parser.add_argument("--max_fraction", type=float, default=0.5, help="Maximum fraction of points to initialize solution.")
    parser.add_argument("--max_iterations", type=int, default=10_000_000, help="Maximum number of local search iterations.")
    parser.add_argument("--max_runtime", type=float, default=60*60, help="Maximum runtime in seconds for local search.")
    parser.add_argument("--doubleswap_time_threshold", type=float, default=60.0, help="Time threshold in seconds after which double swap moves will no longer be considered.")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes to use for local search.")
    parser.add_argument("--output_folder", type=str, help="Path to output folder.")
    args = parser.parse_args()

    # Fetch data
    seq2lin = read_metadata(args.metadata)
    sequence_mapping = read_sequence_mapping(args.sequences_mapping)
    unique_lineages = sorted(list(set(seq2lin.values())))
    clusters = []
    for idx, seq_id in sequence_mapping.items():
        lineage = seq2lin[seq_id]
        clusters.append(unique_lineages.index(lineage))
    clusters = np.array(clusters, dtype=np.int32)

    # Create distance generator
    output_filename = ""
    if args.mash:
        distance_generator = generate_distances_mash(args.distances)
        output_filename = f"MASH_{args.distances.split('_')[-1].replace('.dist', '')}.txt"
    else:
        distance_generator = generate_distances_sourmash(args.distances, jaccard=args.jaccard)
        if args.jaccard:
            output_filename = f"SOURMASH_Jaccard_{args.distances.split('_')[-1].replace('.csv', '')}.txt"
        else:
            output_filename = f"SOURMASH_Cosine_{args.distances.split('_')[-1].replace('.csv', '')}.txt"

    # Initialize solution object
    try:
        start = time.time() # Measure time to initialize solution object
        if args.num_processes <= 1:
            S = Solution.generate_random_solution(
                distances=distance_generator,
                clusters=clusters,
                selection_cost=args.selection_cost,
                cost_per_cluster=0, #use standard cost per cluster (cost for selecting is equal to selection cost)
                max_fraction=args.max_fraction,
                seed=args.seed
            )
        else:
            S = Solution_shm.generate_random_solution(
                distances=distance_generator,
                clusters=clusters,
                selection_cost=args.selection_cost,
                cost_per_cluster=0, #use standard cost per cluster (cost for selecting is equal to selection cost)
                max_fraction=args.max_fraction,
                seed=args.seed
            )
        print("Time spent initializing solution:", time.time() - start, flush=True)

        # Run local search
        start = time.time()
        if args.num_processes <= 1:
            time_per_iteration, objectives = S.local_search(
                max_iterations=args.max_iterations,
                max_runtime=args.max_runtime,
                random_move_order=True, random_index_order=True,
                doubleswap_time_threshold=args.doubleswap_time_threshold,
                logging=True,
                logging_frequency=100,
            )
        else:
            time_per_iteration, objectives = S.local_search(
                num_processes = args.num_processes,
                max_iterations=args.max_iterations,
                max_runtime=args.max_runtime,
                random_move_order=True, random_index_order=True,
                doubleswap_time_threshold=args.doubleswap_time_threshold,
                mp_switch_threshold=15.0,
                logging=True,
                logging_frequency=100,
            )
        print("Time spent in local search:", time.time() - start, flush=True)

        # Sanity check, verify solution is still feasible and if local search solutions a monotonically improving objective
        assert S.determine_feasibility(), "Final solution is infeasible!"
        for i in range(1, len(objectives)):
            assert objectives[i] <= objectives[i-1] + PRECISION_THRESHOLD, "Objective did not monotonically decrease during local search!"

        selected_points = np.copy(S.selection)
        # Output
        S.calculate_objective() #ensure objective is up to date
        if args.num_processes <= 1:
            print("Final objective:", S.objective, flush=True)
        else:
            print("Final objective:", S.objective[0], flush=True)
            S.cleanup()
        print("Number of selected points:", np.sum(selected_points), flush=True)

        # Iterate over selected sequences on a per-lineage basis
        for lineage in unique_lineages:
            lineage_indices = [idx for idx, seq_id in sequence_mapping.items() if seq2lin[seq_id] == lineage]
            num_selected_in_lineage = np.sum(selected_points[lineage_indices])
            if num_selected_in_lineage > 0:
                print(f"Lineage {lineage}: {num_selected_in_lineage} selected sequences", flush=True)
            else:
                raise ValueError(f"No sequences selected for lineage {lineage}!")

        # Write to output file
        if output_filename != "" and args.output_folder is not None:
            for lineage in unique_lineages:
                os.makedirs(f"{args.output_folder}/{lineage}", exist_ok=True)
                lineage_indices = [idx for idx, seq_id in sequence_mapping.items() if seq2lin[seq_id] == lineage]
                with open(f"{args.output_folder}/{lineage}/{output_filename}", "w") as f_out:
                    for idx in lineage_indices:
                        if selected_points[idx]:
                            f_out.write(f"{sequence_mapping[idx]}\n")

        if isinstance(S, Solution_shm):
            S.cleanup()
        del S

    except Exception as e:
        print("An error occurred during execution:", flush=True)
        print(str(e), flush=True)
        traceback.print_exc()

        if isinstance(S, Solution_shm):
            S.cleanup()
        del S

if __name__ == "__main__":
    main()