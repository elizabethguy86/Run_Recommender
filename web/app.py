import numpy as np
import pandas as pd
import pymongo
from stravalib import Client
from model.Filter_Data import Run_Recommender
from model.Group_Runs import GroupRuns
import Mapping_Functions.Map_Routes as mapfun
from io import BytesIO
import folium 

from flask import Flask, request, render_template

mc = pymongo.MongoClient() #setup mongo client to store tokens for approved users
db = mc['runpaw']
tokens = db['strava_tokens']

with open('redirect_uri.txt') as f:
    REDIRECT_URI = f.read().strip()
with open('.strava_client_secret') as f:
    STRAVA_CLIENT_SECRET = f.read().strip()
with open('.strava_client_id') as f:
    STRAVA_CLIENT_ID = f.read().strip()

app = Flask(__name__, static_url_path="")

#dataframe to pull from
data = pd.read_csv('Sample_Data/run_data_11-27-2018.csv')
df = data.iloc[:,1::]

#coordinates dictionary
coords = {}

#authorize new users
def get_auth_url():
    """Get the Strava authorization URL."""
    client = Client()
    auth_url = client.authorization_url(
        client_id=STRAVA_CLIENT_ID,
        redirect_uri= REDIRECT_URI)
    return auth_url

#authorization url
AUTH_URL = get_auth_url()

#landing page
@app.route('/')
def hello_world():
    return render_template('index.html', auth_url=AUTH_URL)

#authorization page
@app.route('/authorize')
def authorize():
    code = request.args.get('code')
    client = Client()
    access_token = client.exchange_code_for_token(client_id=STRAVA_CLIENT_ID,
						  client_secret=STRAVA_CLIENT_SECRET,
						  code=code)
    tokens.insert_one({'token': access_token})
    return render_template('success.html', token=access_token)

#error page
@app.errorhandler(500)
def page_not_found(e):
    script = '<p class="text-white-50">No runs in this area.  Please try again.</p>'
    return render_template('index.html', message=script)

#return recommendations
@app.route('/request', methods=['POST'])
def make_recommendations():
    '''Takes in user inputs from website and returns run recommendations
    with stats on each recommended route and the map of the routes.'''
    if not request.form['user_input_location']:
        script = '<p class="text-white-50">Please enter a location.</p>'
        return render_template('index.html', message=script, auth_url=AUTH_URL)
    elif not request.form['user_input']:
        script = '<p class="text-white-50">Please enter your preferred elevation gain and distance.</p>'
        return render_template('index.html', message=script, auth_url=AUTH_URL)
    elif not request.form['input_dist']:
        script = '<p class="text-white-50">Please enter the preferred distance you are willing to travel.</p>'
        return render_template('index.html', message=script, auth_url=AUTH_URL)
    location = request.form['user_input_location']
    location = location.split(',')
    #change from txt to float
    location = (float(location[0]), float(location[1])) 
    recommendations = Run_Recommender(df, location)
    req = request.form['user_input']
    req = req.split(',')
    #convert feet to meters
    req = [float(req[0])*0.3048, float(req[1])] 
    dist = float(request.form['input_dist'])
    recommend_dict, similarities = recommendations.recommend_runs(req, dist)
    polylines, indices = recommendations.make_polyline_dict()
    Group = GroupRuns(polylines, indices, df)
    map_coordinates = Group.map_coordinates()
    indices_to_use = Group.make_groups(threshold=0.05)
    unique_coordinates = [map_coordinates[i] for i in indices_to_use]
    query = (location, tuple(req))
    query_id = hash(query)
    coords[query_id] = unique_coordinates
    mapping_dict = mapfun.map_indices(indices_to_use, indices)
    stats_df = mapfun.return_route_stats(mapping_dict, indices_to_use, df)
    mapping = map_runs(unique_coordinates)
    i_frame = '<iframe src="/map/' + str(query_id) + '" width="1000" height="500"> </iframe>'
    return render_template('index.html', table = stats_df.to_html(classes=''),
                             map=i_frame, auth_url=AUTH_URL)

#map rendering
@app.route('/map/<query_id>', methods=['GET'])
def map(query_id):
    '''looks up the coordinates in the global dictionary based on 
    the query_id key. Key is derived from the user inputs of location
    and the request (see make_recommendations()).'''
    query_id = int(query_id)
    unique_coordinates = coords[query_id]
    return map_runs(unique_coordinates)

#map creation
def map_runs(unique_coordinates):
    '''Takes in a list of coordinates for running routesand returns the html for the 
    folium map with routes plotted in different colors.'''
    #get start point for the map
    print("unique_coordinates[0][0]: ", unique_coordinates[0][0]) #checking outputs
    lat, long = unique_coordinates[0][0]
    m = folium.Map(location=[lat, long], zoom_start=12.3)
    for idx, route in enumerate(unique_coordinates[0:5]):
        colors = ['blue','green','red','orange','purple']
        folium.PolyLine(
                route,
                weight=2,
                color=colors[idx]
            ).add_to(m)
    #create legend for colors to route number
    legend_html = '''<div style= "position: fixed; 
        bottom: 50px; left: 50px; width: 100px; height: 180px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        ">&nbsp; Routes <br>
        &nbsp; Route 1 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:blue"></i><br>
        &nbsp; Route 2 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:green"></i>
        &nbsp; Route 3 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:red"></i><br>
        &nbsp; Route 4 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:orange"></i>
        &nbsp; Route 5 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:purple"></i>
        </div>'''
    m.get_root().html.add_child(folium.Element(legend_html)) #add legend to map
    mapdata = BytesIO()
    m.save(mapdata, close_file=False)
    html = mapdata.getvalue()
    return html