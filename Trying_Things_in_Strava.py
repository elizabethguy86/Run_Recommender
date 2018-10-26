from stravalib.client import Client
import json
import pandas as pd
TOKEN = token
client_me = Client(access_token=TOKEN)


'''Start Flask'''

FLASK_APP=app.py flask run

'''Get tokens from MongoDB'''
tmux
db.strava_tokens.find()

activities = client_me.get_activities(limit=1000)
#looking at the columns I'm interested in
sample = list(activities)[0]
my_cols =  ['average_speed',
          'distance',
          'elapsed_time',
          'total_elevation_gain',
          'type',
          'start_date_local',
          'start_latlng',
          'start_longitude',
          ]

'''use sample of data to inform choice of columns above'''
sample.to_dict()
data = []
for activity in activities:
    my_dict = activity.to_dict()
    data.append([my_dict.get(x) for x in my_cols])

'''Create DataFrame for the colummns of interest'''
df = pd.DataFrame(data, columns=my_cols)

'''Convert 'distance' to readable miles'''
df['miles_converted'] = [x/1609.3440122044242 for x in df['distance']]


streams = client_me.get_activity_streams(123, types=types, resolution='medium')
routes = client_me.get_routes(athlete_id=id)

#  Result is a dictionary object.  The dict's key are the stream type.
if 'distance' in streams.keys():
    print(streams['distance'].data)

'''Get list of users and their tokens from MongoDB'''
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
        if item not in unique:
            unique.append(item)

    return unique


'''Draw activity Map'''
#get coordinates from activity first
activity = client.get_activity(activity_id)

activity.to_dict()
#get polyline to decode for activity of interested
activity_poly = activity.to_dict()['map']['summary_polyline']

import polyline
#decode polyline object into coordinates
coordinates = polyline.decode(activity_poly)

#draw some maps
import folium

#get start point for map
lat, long = coordinates[0]
map = folium.Map(location=[lat, long], zoom_start=12.2)

folium.PolyLine(
        coordinates,
        weight=2,
        color='blue'
    ).add_to(map)
map
