import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale
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
        self.df = df
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
        lat1, lon1 = coordinate1
        lat2, lon2 = coordinate2
        distance = self.haversine(lat1, lon1, lat2, lon2)
        return distance

    def make_floats(self, tup):
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

    # latitude/longitude start location
    # start = (47.529832, -121.987695)

    def standardize_inputs(self, user_input, df):
        '''Standardize the user inputs for cosine similarity'''
        elevation = user_input[0]
        distance = user_input[1]
        std_elevation = (elevation - df['total_elevation_gain'].mean())/df['total_elevation_gain'].std()
        std_distance = (distance - df['miles_converted'].mean())/df['miles_converted'].std()
        return np.array([std_elevation,std_distance])

    def recommend_runs(self, request, dist):
        '''Inputs are a list of user-specified elevation gain in meters and miles to run, 
        and distance away for willingness to travel to a run.  
        Output is a dictionary of polyline maps for route recommendations.'''
        df = self.get_distances()
        self.request = request
        self.dist = dist 
        #requires sklearn.cosine_similarity
        df = df[df['distance_away']<= self.dist] #filter dataframe for the requested distance range
        df.loc[:, 'elevation_std'] = scale(df['total_elevation_gain'].values.reshape(-1, 1))
        df.loc[:, 'miles_std'] = scale(df['miles_converted'].values.reshape(-1, 1))
        similarity_df = df.loc[:, ['elevation_std', 'miles_std']]
        user_input = self.standardize_inputs(self.request, df)
        user_input = user_input.reshape(1,2)
        user_input_reshaped = user_input.reshape(1,-1)
        similarities = cosine_similarity(similarity_df, user_input_reshaped)
        sort_indices = np.argsort(similarities, axis = None)
        top_20 = sort_indices[-20:]
        recommend_indices = list(top_20[::-1]) #reverse the order
        recommendations = df.iloc[recommend_indices, :]
        return dict(recommendations['map'])

    def make_polyline_dict(self):
        '''Take in a dictionary of map objects and return dictionary of polylines{index:polyline} and the indices
        for the polylines as a list.'''
        recommend_dict = self.recommend_runs(self.request, self.dist)
        polylines = {}
        for k, v in recommend_dict.items():
            v = ast.literal_eval(v)
            if v['summary_polyline'] != None: #make sure the polyline list isn't empty
                polylines[k] = v['summary_polyline']
        indices = list(polylines.keys())
        return polylines, indices