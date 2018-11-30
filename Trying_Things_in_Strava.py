from stravalib.client import Client
from stravalib import model
import json
import pandas as pd
import numpy as np
import polyline
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale
from math import radians, cos, sin, asin, sqrt


'''Start Flask'''

FLASK_APP=app.py flask run

'''Get tokens from MongoDB'''
tmux
db.strava_tokens.find()

'''Remap localhost to EC2 server'''
! ssh -NL 47017:localhost:27017 quesadilla

'''Get list of users and their tokens from MongoDB.
Instructions: Set up tunnel ssh ssh -NL 47017:localhost:27017 #your EC2 name'''
mc = MongoClient(host='localhost:47017')
users = mc['runpaw']
tokens = list(users.strava_tokens.find())

def make_tokens_list(l):
    '''Take a list of users from MongoDB and extract their
       tokens from the dictionary objects in the list'''
    tokens_list = []
    unique = []
    for token in tokens:
        tokens_list.append(token['token'])
    for item in tokens_list:
        if item not in unique: #if someone double-clicks, you get copies of the token
            unique.append(item)

    return unique

'''Decide on attributes to use'''
activities = client_me.get_activities(limit=1000)
#looking at the columns I'm interested in
sample = list(activities)[0]
cols = ['upload_id',
        'average_speed',
        'distance',
        'elapsed_time',
        'total_elevation_gain',
        'type',
        'start_date_local',
        'start_latlng',
        'end_latlng', 'map']

'''use sample of data to inform choice of columns above'''
sample.to_dict()

'''Make dataframe of all activities'''
def activities_to_dict(tokens):
    data = []
    for token in tokens:
        client_current = Client(access_token=token)
        activities = client_current.get_activities(limit=1000)
        for activity in activities:
            my_dict = activity.to_dict()
            data.append([my_dict.get(x) for x in cols])
        #make large dataframe for columns of interest for all tokens
    df = pd.DataFrame(data, columns=cols)
    #convert distance to miles
    df['miles_converted'] = [x/1609.3440122044242 for x in df['distance']]
    return df

'''Create DataFrame for the colummns of interest'''
df = pd.DataFrame(data, columns=cols)

'''Convert 'distance' to readable miles'''
df['miles_converted'] = [x/1609.3440122044242 for x in df['distance']]

'''Only use runs'''
df = df[df['type'] == 'Run']

streams = client_me.get_activity_streams(123, types=types, resolution='medium')
routes = client_me.get_routes(athlete_id=id)

#  Result is a dictionary object.  The dict's key are the stream type.
if 'distance' in streams.keys():
    print(streams['distance'].data)

#filter datframe by proximity

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).

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
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def find_distances(coordinate1, coordinate2):
    lat1, lon1 = coordinate1
    lat2, lon2 = coordinate2
    distance = haversine(lat1, lon1, lat2, lon2)
    return distance


def make_floats(tup):
    x, y = tup
    return (float(x), float(y))


#make distance away column. Start is a user input latlng coordinate in tuple form
def get_distances(df, start):
    '''Takes in a dataframe of activities and returns a dataframe with start and end latlng as tuples with floats.
    Also returns the distance away from a starting point input.'''
   
    df_starts = df[(~df['start_latlng'].isna()) & (~df['end_latlng'].isna())].copy()
    df_starts['start_latlng'] = df_starts['start_latlng'].apply(lambda x: x.split(","))
    df_starts['start_latlng'] = df_starts['start_latlng'].apply(lambda x: tuple(x))
    df_starts['start_latlng'] = df_starts['start_latlng'].apply(lambda x: make_floats(x))
    df_starts['end_latlng'] = df_starts['end_latlng'].apply(lambda x: x.split(","))
    df_starts['end_latlng'] = df_starts['end_latlng'].apply(lambda x: tuple(x))
    df_starts['end_latlng'] = df_starts['end_latlng'].apply(lambda x: make_floats(x))
    df_starts['distance_away'] = df_starts['start_latlng'].apply(lambda x: find_distances(start, x))
    return df_starts
    

# For cosine similarity.  Standardize user inputs according to matching data columns
# in dataframe.
def standardize_inputs(user_input, df):
    '''Standardize the user inputs for cosine similarity comparison'''
    elevation = user_input[0]
    distance = user_input[1]
    std_elevation = (elevation - df['total_elevation_gain'].mean())/df['total_elevation_gain'].std()
    std_distance = (distance - df['miles_converted'].mean())/df['miles_converted'].std()
    return np.array([std_elevation,std_distance])

#standardize the columns of interest for your dataframe
similarity_df['elevation_std'] = scale(similarity_df['total_elevation_gain'])
similarity_df['miles_std'] = scale(similarity_df['miles_converted'])


# Function to recommmend runs
def recommend_runs(request, df, columns_to_check):
    '''Inputs are a list of user-specified elevation gain in meters and miles to run, 
    dataframe of activities, and the columns of the dataframe to check 
    for cosine similarity. Columns to check should be in standardized form.  
    Output is a dictionary of polyline maps for route recommendations.'''
    df = df.copy()
    #requires sklearn.cosine_similarity
    df.loc[:, 'elevation_std'] = scale(df['total_elevation_gain'].values.reshape(-1, 1))
    df.loc[:, 'miles_std'] = scale(df['miles_converted'].values.reshape(-1, 1))
    similarity_df = df.loc[:, columns_to_check]
    user_input = standardize_inputs(request, df)
    user_input = user_input.reshape(1,len(columns_to_check))
    user_input_reshaped = user_input.reshape(1,-1)
    similarities = cosine_similarity(similarity_df, user_input_reshaped)
    sort_indices = np.argsort(similarities, axis = None)
    top_20 = sort_indices[-20:]
    recommend_indices = list(top_20[::-1]) #reverse the order
    recommendations = df.iloc[recommend_indices, :]
    return dict(recommendations['map']), recommend_indices

'''Extract polyline from dataframe'''
one_point = dict(df.iloc[idx,map_column]
summary_polyline = one_point['summary_polyline']

'''Draw activity Map'''
#get coordinates from activity first
activity = client.get_activity(activity_id)

activity.to_dict()
#get polyline to decode for activity of interested
activity_poly = activity.to_dict()['map']['summary_polyline']

# Make a list of polylines from the recommmendations dictionary
def make_polyline_dict(recommend_dict):
    '''Take in a dictionary of map objects and return dictionary of polylines{index:polyline} and the indices
    for the polylines as a list.'''
    polylines = {}
    for k, v in recommend_dict.items():
        v = ast.literal_eval(v)
        if v['summary_polyline'] != None: #make sure the polyline list isn't empty
            polylines[k] = v['summary_polyline']
    indices = list(polylines.keys())
    return polylines, indices

#make list of coordinate lists
polylines = make_polyline_dict(recommendations)
map_coordinates = []
for line in polylines:
    coordinates = polyline.decode(line)
    map_coordinates.append(coordinates)

#make unique route list

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

comparison_df = make_comparison_df(map_coordinates, working_df, indices)
comparison_array = comparison_df.values
comparison_array_std = (comparison_array - np.mean(comparison_array, axis=0)) / np.std(comparison_array, axis=0)
#make comparisons with all the datapoints in the comparison array
cosine_sim_arr = cosine_similarity(comparison_array_std)

#Use Hierarchical Clustering to find groupings for the routes
threshold = 0.05
Z = hierarchy.linkage(cosine_sim_arr, 'average', metric="cosine")
C = hierarchy.fcluster(Z, threshold, criterion="distance")
ids = list(range(20)) #make a list of numbers 0-19 to use as indices for a cluster groups dictionary
C

cluster_groups = defaultdict(list)
for idx, grouping in enumerate(C):
    cluster_groups[grouping].append(idx)
def get_indices(groups):
    '''Takes in dictionary of cluster_groupings and returns a list of indices
    to use for route suggestions'''
    sort_groups = sorted(list(groups.values()), key=len)
    sort_groups = sort_groups[::-1]
    indices_to_use = []
    for group in sort_groups:
        if len(group) >= 1:
            indices_to_use.append(group[0])
    return indices_to_use

indices_to_use = get_indices(cluster_groups)

unique_coordinates = [map_coordinates[i] for i in indices_to_use]

def map_indices(indices_to_use, indices):
    '''Takes in indices_to_use from 20 suggested routes and the actual indices of the 20 routes in the larger
    dataframe and returns a mapping of indices_to_use back to the index in the larger dataframe. 
    Use: to retrieve stats for suggested routes'''
    mapping = {}
    for idx, i in enumerate(indices):
        if idx in indices_to_use:
            mapping[idx] = i
    return mapping

#use for later route stats lookups
mapping_dict = map_indices(indices_to_use, indices)

'''obsolete?'''
# def return_unique_idx(map_coordinates):
#     '''return indices for the unique maps in a list of coordinates'''
#     cent_dict = {}
#     centroids = find_centroids(map_coordinates)
#     for idx, c in enumerate(centroids):
#         cent_dict[c] = idx
#     return list(cent_dict.values())

# indices = return_unique_idx(map_coordinates)
# unique_coordinates = [map_coordinates[i] for i in indices]

