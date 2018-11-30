import pandas as pd
import numpy as np 
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity




def find_centroids(coordinate_lst):
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

def make_comparison_df(coordinate_lst, df, indices):
    centroids = find_centroids(coordinate_lst)
    lats = []
    longs = []
    elevation_lst = []
    for c in centroids:
        lats.append(c[0])
        longs.append(c[1])
    for idx in indices: #get the elevation for the runs in the suggestion list.
        row = df.loc[idx] 
        elevation_lst.append(row['total_elevation_gain'])
    comparison_df = pd.DataFrame({'lats': lats, 'longs':longs, 'elevation':elevation_lst})
    return comparison_df

#make a dataframe of centroid coordinates(lat, long) and elevation gain for
#later comparisons of route similarity
comparison_df = make_comparison_df(map_coordinates, working_df, indices)

def cosine_sim_threshold_idx(df):
    '''Takes in a dataframe of map lat centroids, map long centroids, and elevation gain and returns a list
       of indices that are below a similarity threshold for each map'''
    comparison_array = df.values
    comparison_array_std = (comparison_array - np.mean(comparison_array, axis=0)) / np.std(comparison_array, axis=0)
    #make comparisons with all the datapoints in the comparison array
    cosine_sim_arr = cosine_similarity(comparison_array_std)
    idx_dict = {}
    for idx1, sim in enumerate(cosine_sim_arr): 
        #goes through cosine similarity arrays for each index 
        #and returns those that are below a similarity threshold
        lst = []
        for idx, num in enumerate(sim):
            if abs(num) > 0.993:
                lst.append(idx)
        idx_dict[idx1] = tuple(lst)
    groups = [set(x) for x in set(idx_dict.values())]
    unique_groups = merge_groups(groups) #this needs to be fixed!!!!
    return unique_groups, groups

unique_groups, groups = cosine_sim_threshold_idx(comparison_df)

def get_indices(groups):
    '''Take in a list of index groupings and return one item from each of the
    groupings, prioritizing taking from the largest group first'''
    sort_groups = sorted(groups, key=len)
    sort_groups = sort_groups[::-1]
    indices_to_use = []
    for group in sort_groups:
        group = list(group)
        indices_to_use.append(group[0])
    return indices_to_use

def merge_groups(groups):
    #need to fix the case where there are chains of intersections:  A in B and B in C
    merged_groups = []
    unique_groups = []
    for group1 in groups:
        for group2 in groups:
            if group1 != group2 and group1.intersection(group2):
                merged = group1.union(group2)
                merged_groups.append(merged)
            else:
                merged_groups.append(group1)
    for g in merged_groups:
        if g not in unique_groups:
            unique_groups.append(g)
    return unique_groups