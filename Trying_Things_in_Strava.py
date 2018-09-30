from stravalib.client import Client
import json
import pandas as pd
TOKEN = 'a28defa6db38f9732492b90a7d5faf0109b8166b'
client_me = Client(access_token=TOKEN)
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
          'map']

'''use sample of data to inform columns above'''
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

#  Result is a dictionary object.  The dict's key are the stream type.
if 'distance' in streams.keys():
    print(streams['distance'].data)
