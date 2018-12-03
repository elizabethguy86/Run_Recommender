import numpy as np
import pandas as pd
import polyline
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import  hierarchy

class GroupRuns():
    '''Takes in the original datframe of all runs before the suggestor and the
    polylines dictionary and indices from the Recommend_Runs() class.'''
    
    def __init__(self, polylines, indices, original_df):
        self.polylines = polylines
        self.indices = indices
        self.df = original_df

    def map_coordinates(self):
        #get coordinates for the polylines in the dictionary
        self.map_coordinates = []
        for line in list(self.polylines.values()):
            coordinates = polyline.decode(line)
            map_coordinates.append(coordinates)
        return self.map_coordinates
    
    def find_centroids(self, coordinate_lst):
        centroids = []
        for l in coordinate_lst:
            lats = []
            longs = []
            for point in l:
                lats.append(point[0])
                longs.append(point[1])
            centroid = (round(np.mean(lats), 3), round(np.mean(longs), 3))
            centroids.append(centroid)
        return centroids

    def make_comparison_array(self):
        coordinate_lst = self.map_coordinates()
        indices = self.indices 
        centroids = self.find_centroids(coordinate_lst)
        lats = []
        longs = []
        elevation_lst = []
        for c in centroids:
            lats.append(c[0])
            longs.append(c[1])
        for idx in indices: #get the elevation for the runs in the suggestion list.
            row = self.df.loc[idx] 
            elevation_lst.append(row['total_elevation_gain'])
        comparison_df = pd.DataFrame({'lats': lats, 'longs':longs, 'elevation':elevation_lst})
        comparison_array = comparison_df.values
        comparison_array_std = (comparison_array - np.mean(comparison_array, axis=0)) / np.std(comparison_array, axis=0)
        #make comparisons with all the datapoints in the comparison array
        cosine_sim_arr = cosine_similarity(comparison_array_std)
        return cosine_sim_arr
    
    def make_groups(self, threshold):
        Z = hierarchy.linkage(cosine_sim_arr, 'average', metric="cosine")
        C = hierarchy.fcluster(Z, threshold, criterion="distance")
        cluster_groups = defaultdict(list)
        for idx, grouping in enumerate(C):
            cluster_groups[grouping].append(idx)
        sort_groups = sorted(list(cluster_groups.values()), key=len)
        sort_groups = sort_groups[::-1]
        indices_to_use = []
        for group in sort_groups:
            if len(group) >= 1:
                indices_to_use.append(group[0])
        return indices_to_use