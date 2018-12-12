import numpy as np
import pandas as pd
import polyline
from collections import defaultdict, Counter
from math import radians,cos,sin
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
        map_coordinates = []
        for line in list(self.polylines.values()):
            coordinates = polyline.decode(line)
            map_coordinates.append(coordinates)
        return map_coordinates
    
    def find_centroids(self, coordinate_lst):
        '''Find the centroid lat, long for coordinates in coordinate_lst.'''
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
    
    def cartesian(self, lat, long):
        '''converts circular coordinates to cartesian.'''
        phi = radians(90-lat)
        theta = radians(long)
        r = 6371.0 #radius in km here
        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)
        return x, y, z

    def make_comparison_array(self):
        '''Generates an array for the top recommended runs (rows).  Column features are
        centroid latitude, centroid longitude, and elevation gain.'''
        coordinate_lst = self.map_coordinates()
        indices = self.indices 
        centroids = self.find_centroids(coordinate_lst)
        #convert centroids to cartesian coordinates
        cartesians = [self.cartesian(x, y, z) for x, y in centroids]
        xs = []
        ys = []
        zs = []
        elevation_lst = []
        for c in cartesians:
            xs.append(c[0])
            ys.append(c[1])
            zs.append(c[2])
        for idx in indices: #get the elevation for the runs in the suggestion list.
            row = self.df.loc[idx] 
            elevation_lst.append(row['total_elevation_gain'])
        comparison_df = pd.DataFrame({'lats': xs, 'longs': ys, 'zs': zs, 'elevation':elevation_lst})
        comparison_array = comparison_df.values
        comparison_array_std = (comparison_array - np.mean(comparison_array, axis=0)) / np.std(comparison_array, axis=0)
        #make comparisons with all the datapoints in the comparison array
        cosine_sim_arr = cosine_similarity(comparison_array_std)
        return cosine_sim_arr
    
    def make_groups(self, threshold):
        '''Uses cosine similarity as a distance metric for unsupervised flat clustering.
        Threshold determines stopping point for RSS of each vector to its centroid for all 
        vector/centroid groups. Threshold may be modified.'''
        cosine_sim_arr = self.make_comparison_array()
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