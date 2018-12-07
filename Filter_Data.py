import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import scale, StandardScaler
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict, Counter
from scipy.cluster import  hierarchy

class Run_Recommender():
    '''For the purpose of filtering data according to the user-specified inputs.  Data is
    first filtered by location.  Then, Cosine Similarity is used to identify the most similar
    routes within the specified area based on elevation gain and distance to run.'''

    def __init__(self, df, start):
        '''takes in a dataframe created from Activities class using the 
        activities_to_dict() method'''
        self.df = df[df['miles_converted'] >= 1]
        self.start = start

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        paraphrased from 
        https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 3956 # Radius of earth in miles. Use 6371 for km.
        return c * r

    def find_distances(self, coordinate1, coordinate2):
        '''Find the distance between two coordinates'''
        lat1, lon1 = coordinate1
        lat2, lon2 = coordinate2
        distance = self.haversine(lat1, lon1, lat2, lon2)
        return distance

    def make_floats(self, tup):
        '''makes a tuple of floats. Used for coordinate unpacking'''
        x, y = tup
        return (float(x), float(y))

    def get_distances(self):
        '''Takes in a dataframe of activities and returns a dataframe with start and end latlng as tuples with floats.
        Also returns the distance away from a starting point input.'''
        df = self.df 
        start = self.start
        df_starts = df[(~df['start_latlng'].isna()) & (~df['end_latlng'].isna())].copy()
        df_starts.loc[:,'start_latlng'] = df_starts.loc[:,'start_latlng'].apply(lambda x: x.split(","))
        df_starts.loc[:,'start_latlng'] = df_starts.loc[:,'start_latlng'].apply(lambda x: tuple(x))
        df_starts.loc[:,'start_latlng'] = df_starts.loc[:,'start_latlng'].apply(lambda x: self.make_floats(x))
        df_starts.loc[:,'end_latlng'] = df_starts.loc[:,'end_latlng'].apply(lambda x: x.split(","))
        df_starts.loc[:,'end_latlng'] = df_starts.loc[:,'end_latlng'].apply(lambda x: tuple(x))
        df_starts.loc[:,'end_latlng'] = df_starts.loc[:,'end_latlng'].apply(lambda x: self.make_floats(x))
        df_starts.loc[:,'distance_away'] = df_starts.loc[:,'start_latlng'].apply(lambda x: self.find_distances(start, x))
        return df_starts


    def recommend_runs(self, request, dist):
        '''Inputs are a list of user-specified elevation gain in meters and miles to run
        [meters, miles], and distance away (float) for willingness to travel to a run.  
        Output is a dictionary of polyline maps for route recommendations.'''
        df = self.get_distances()
        self.request = request
        self.dist = dist 
        scaler = StandardScaler()
        #requires sklearn.euclidean_distances()
        df = df[df['distance_away']<= self.dist] #filter dataframe for the requested distance range
        if len(df) == 0:
            raise Exception("No Runs in this area.  Try again with different coordinates")
        else: 
            # df = df[df['distance_away']<= dist] #filter dataframe for the requested distance range
            X = df.loc[:,['total_elevation_gain', 'miles_converted']]
            scaler.fit(X)
            similarity_df = scaler.transform(X)
            user_input_reshaped = scaler.transform(np.array(request).reshape(1,-1))
            similarities = euclidean_distances(similarity_df, user_input_reshaped)
            sort_indices = np.argsort(similarities, axis = None)
            top_20 = sort_indices[0:20]
            recommend_indices = list(top_20) #make it a list for slicing
            recommendations = df.iloc[recommend_indices, :]
            return dict(recommendations['map']), similarities

    def make_polyline_dict(self):
        '''Take in a dictionary of map objects and return dictionary of polylines{index:polyline} and the indices
        for the polylines as a list.'''
        recommend_dict, similarities = self.recommend_runs(self.request, self.dist)
        polylines = {}
        for k, v in recommend_dict.items():
            v = ast.literal_eval(v)
            if v['summary_polyline'] != None: #make sure the polyline list isn't empty
                polylines[k] = v['summary_polyline']
        indices = list(polylines.keys())
        return polylines, indices