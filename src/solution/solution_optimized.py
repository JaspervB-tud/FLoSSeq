import numpy as np
import itertools
import math

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
        self.distances = distances.copy()
        self.clusters = clusters.copy()
        self.unique_clusters = np.unique(self.clusters)
        self.selection_cost = selection_cost

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
                cur_min = np.inf
                cur_idx = idx
                for other_idx in np.where((self.clusters == self.clusters[idx]) & self.selection)[0]:
                    cur_dist = self.distances[idx, other_idx]
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
                cur_max = -np.inf
                cur_pair = (None, None)
                for point_pair in itertools.product(cluster_1, cluster_2):
                    cur_dist = 1 - self.distances[point_pair[0], point_pair[1]]
                    if cur_dist > cur_max:
                        cur_max = cur_dist
                        cur_pair = point_pair
                self.closest_distances_inter[cluster_pair[0], cluster_pair[1]] = cur_max
                self.closest_distances_inter[cluster_pair[1], cluster_pair[0]] = cur_max
                self.closest_points_inter[(cluster_pair[0], cluster_pair[1])] = cur_pair  # Store the first point index
                self.objective += cur_max
        else:
            # If selection provided is not feasible, don't do anything (YET)
            self.objective = np.inf
            print("The solution is infeasible, objective value is set to INF and closest distances & points are not set.")

    def evaluate_add(self, idx_to_add):
        """
        Evaluates whether the proposed addition improves the current solution.
        """
        if self.selection[idx_to_add]:
            raise ValueError("The point to add must not be selected.")
        cluster = self.clusters[idx_to_add]
        candidate_objective = self.objective + self.selection_cost # cost for adding the point
        # Calculate intra cluster distances for cluster of new point
        add_within_cluster = [] #this stores changes that have to be made if the objective improves
        for idx in self.nonselection_per_cluster[cluster]:
            cur_dist = self.distances[idx, idx_to_add] # distance to current point (idx)
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
                    cur_similarity = 1 - self.distances[idx, idx_to_add] #this is the similarity, if it is more similar then change solution
                    if cur_similarity > cur_max:
                        cur_max = cur_similarity
                        cur_idx = idx
                if cur_idx > -1:
                    candidate_objective += cur_max - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((cluster, other_cluster, cur_max, cur_idx))

        return candidate_objective, add_within_cluster, add_for_other_clusters
    
    def evaluate_swap(self, idx_to_add, idx_to_remove):
        """
        Evaluates whether the proposed swap improves the current solution.
        """
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
                    cur_dist = self.distances[idx, other_idx]
                    if cur_dist < cur_closest_distance:
                        cur_closest_distance = cur_dist
                        cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
            else: #point to be removed is not closest, check if newly added point is closer
                cur_dist = self.distances[idx, idx_to_add]
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
                            cur_similarity = 1 - self.distances[idx, other_idx]
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                if other_cluster < cluster:
                                    cur_closest_pair = (idx, other_idx)
                                else:
                                    cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((cluster, other_cluster, cur_closest_pair, cur_closest_similarity))
                else: #point to be removed is not closest, check if newly added point is closer
                    for idx in self.selection_per_cluster[other_cluster]:
                        cur_similarity = 1 - self.distances[idx, idx_to_add]
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            if other_cluster < cluster:
                                cur_closest_pair = (idx, idx_to_add)
                            else:
                                cur_closest_pair = (idx_to_add, idx)
                    if cur_closest_pair[0] > -1:
                        candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                        add_for_other_clusters.append((cluster, other_cluster, cur_closest_pair, cur_closest_similarity))

        return candidate_objective, add_within_cluster, add_for_other_clusters

    def evaluate_doubleswap(self, idxs_to_add, idx_to_remove):
        """
        Evaluates whether the proposed double swap improves the current solution.
        """
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
                    cur_dist = self.distances[idx, other_idx]
                    if cur_dist < cur_closest_distance:
                        cur_closest_distance = cur_dist
                        cur_closest_point = other_idx
                candidate_objective += cur_closest_distance - self.closest_distances_intra[idx]
                add_within_cluster.append((idx, cur_closest_point, cur_closest_distance))
            else: #point to be removed is not closest, check if one of newly added points is closer
                cur_dist1 = self.distances[idx, idx_to_add1]
                cur_dist2 = self.distances[idx, idx_to_add2]
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
                            cur_similarity = 1 - self.distances[idx, other_idx]
                            if cur_similarity > cur_closest_similarity:
                                cur_closest_similarity = cur_similarity
                                if other_cluster < cluster:
                                    cur_closest_pair = (idx, other_idx)
                                else:
                                    cur_closest_pair = (other_idx, idx)
                    candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                    add_for_other_clusters.append((cluster, other_cluster, cur_closest_pair, cur_closest_similarity))
                else: #point to be removed is not closest, check if newly added points are closer
                    for idx in self.selection_per_cluster[other_cluster]:
                        cur_similarity1 = 1 - self.distances[idx, idx_to_add1]
                        cur_similarity2 = 1 - self.distances[idx, idx_to_add2]
                        cur_similarity, idx_to_add = max((cur_similarity1, idx_to_add1), (cur_similarity2, idx_to_add2), key = lambda x: x[0])
                        if cur_similarity > cur_closest_similarity:
                            cur_closest_similarity = cur_similarity
                            if other_cluster < cluster:
                                cur_closest_pair = (idx, idx_to_add)
                            else:
                                cur_closest_pair = (idx_to_add, idx)
                    if cur_closest_pair[0] > -1:
                        candidate_objective += cur_closest_similarity - self.closest_distances_inter[cluster, other_cluster]
                        add_for_other_clusters.append((cluster, other_cluster, cur_closest_pair, cur_closest_similarity))

        return candidate_objective, add_within_cluster, add_for_other_clusters

    def determine_feasibility(self):
        uncovered_clusters = set(self.unique_clusters)
        for point in np.where(self.selection)[0]:
            uncovered_clusters.discard(self.clusters[point])
        return len(uncovered_clusters) == 0