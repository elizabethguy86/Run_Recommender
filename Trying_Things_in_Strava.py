from stravalib.client import Client
import json
import pandas as pd
from pymongo import MongoClient



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
my_cols = ['upload_id',
          'average_speed',
          'distance',
          'elapsed_time',
          'total_elevation_gain',
          'type',
          'start_date_local',
          'start_latlng',
          'start_longitude', 'map']

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
            data.append([my_dict.get(x) for x in my_cols])
        #make large dataframe for columns of interest for all tokens
    return pd.DataFrame(data, columns=my_cols)

'''Create DataFrame for the colummns of interest'''
df = pd.DataFrame(data, columns=my_cols)

'''Convert 'distance' to readable miles'''
df['miles_converted'] = [x/1609.3440122044242 for x in df['distance']]


streams = client_me.get_activity_streams(123, types=types, resolution='medium')
routes = client_me.get_routes(athlete_id=id)

#  Result is a dictionary object.  The dict's key are the stream type.
if 'distance' in streams.keys():
    print(streams['distance'].data)


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
def recommend_runs(user_input, df, columns_to_check):
    '''Inputs are a list of user-specified elevation gain in meters and miles to run, 
    dataframe of activities, and the columns of the dataframe to check 
    for cosine similarity. Columns to check should be in standardized form.  
    Output is a dataframe of route recommendations.'''
    
    #requires sklearn.cosine_similarity
    
    similarity_df = df.loc[:, columns_to_check]
    user_input = standardize_inputs(user_input, df)
    user_input = user_input.reshape(1,len(columns_to_check))
    user_input_reshaped = user_input.reshape(1,-1)
    similarities = cosine_similarity(similarity_df, user_input_reshaped)
    sort_indices = np.argsort(similarities, axis = None)
    top_5 = sort_indices[-5:]
    recommend_indices = list(top_5[::-1]) #reverse the order
    recommendations = df.iloc[recommend_indices, :]
    return recommendations

'''Extract polyline from dataframe'''
one_point = dict(df.iloc[idx,map_column]
summary_polyline = one_point['summary_polyline']

'''Draw activity Map'''
#get coordinates from activity first
activity = client.get_activity(activity_id)

activity.to_dict()
#get polyline to decode for activity of interested
activity_poly = activity.to_dict()['map']['summary_polyline']

# Make a list of polylines from the recommmendations df
def make_polyline_lst(recommendations):
    '''takes in column of map data and returns a list of polylines'''
    recommend_dict = dict(recommendations['map'])
    polylines = []
    for k, v in recommend_dict.items():
        polylines.append(v['summary_polyline'])
    return polylines

import polyline
#decode polyline object into coordinates
coordinates = polyline.decode(activity_poly)

#draw some maps
import folium

#get start point for map
lat, long = coordinates[0]
m = folium.Map(location=[lat, long], zoom_start=12.2)

for idx, route in enumerate(map_coordinates):
    '''Map all routes in different colors'''
    colors = ['blue','green','red','orange','purple']
    folium.PolyLine(
            route,
            weight=2,
            color=colors[idx]
        ).add_to(m)
m