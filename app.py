import numpy as np
import pandas as pd
from Filter_Data import Run_Recommender
from Group_Runs import GroupRuns
import Mapping_Functions.Map_Routes as mapfun
from io import BytesIO
import folium 

from flask import Flask, request, render_template, jsonify


app = Flask(__name__, static_url_path="")

#dataframe to pull from
data = pd.read_csv('Sample_Data/run_data_11-27-2018.csv')
df = data.iloc[:,1::]

#coordinates dictionary
coords = {}

@app.route('/') #landing page
def hello_world():
    return render_template('index.html')

@app.route('/request', methods=['POST'])
def make_recommendations():
    '''Takes in user inputs from website and returns run recommendations
    with stats on each recommended route and the map of the routes.'''
    location = request.form['user_input_location']
    location = location.split(',')
    location = (float(location[0]), float(location[1]))
    recommendations = Run_Recommender(df, location)
    #recommendations = Run_Recommender(df, (47.508802, -122.464284))
    req = request.form['user_input']
    req = req.split(',')
    req = [float(req[0]), float(req[1])]
    recommend_dict, similarities = recommendations.recommend_runs(req, 5)
    polylines, indices = recommendations.make_polyline_dict()
    Group = GroupRuns(polylines, indices, df)
    map_coordinates = Group.map_coordinates()
    indices_to_use = Group.make_groups(threshold=0.05)
    unique_coordinates = [map_coordinates[i] for i in indices_to_use]
    query = (location, tuple(req))
    query_id = hash(query)
    coords[query_id] = unique_coordinates
    mapping_dict = mapfun.map_indices(indices_to_use, indices)
    stats = mapfun.return_route_stats(mapping_dict, indices_to_use, df)
    abbrev_stats = stats.loc[:, ['total_elevation_gain', 'miles_converted']].values
    stats_df = pd.DataFrame(abbrev_stats, columns=['elevation gain', 'miles'])
    mapping = map_runs(unique_coordinates)
    i_frame = '<iframe src="/map/' + str(query_id) + '" width="1000" height="500"> </iframe>'
    return render_template('index.html', table = stats_df.to_html(classes=''), map=i_frame)

# def table_practice():
#     recommendations = Run_Recommender(df, (47.508802, -122.464284))
#     recommend_dict, similarities = recommendations.recommend_runs([200,15], 5)
#     polylines, indices = recommendations.make_polyline_dict()
#     Group = GroupRuns(polylines, indices, df)
#     map_coordinates = Group.map_coordinates()
#     indices_to_use = Group.make_groups(threshold=0.05)
#     unique_coordinates = [map_coordinates[i] for i in indices_to_use]
#     mapping_dict = mapfun.map_indices(indices_to_use, indices)
#     stats = mapfun.return_route_stats(mapping_dict, indices_to_use, df)
#     abbrev_stats = stats.loc[:, ['total_elevation_gain', 'miles_converted']]
#     return abbrev_stats.to_html(), unique_coordinates


@app.route('/map/<query_id>', methods=['GET'])
def map(query_id):
    '''looks up the coordinates in the global dictionary based on 
    the query_id key. Key is derived from the user inputs of location
    and the request (see make_recommendations()).'''
    query_id = int(query_id)
    unique_coordinates = coords[query_id]
    return map_runs(unique_coordinates)

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
        &nbsp; Route 0 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:blue"></i><br>
        &nbsp; Route 1 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:green"></i>
        &nbsp; Route 2 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:red"></i><br>
        &nbsp; Route 3 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:orange"></i>
        &nbsp; Route 4 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:purple"></i>
        </div>'''
    m.get_root().html.add_child(folium.Element(legend_html)) #add legend to map
    mapdata = BytesIO()
    m.save(mapdata, close_file=False)
    html = mapdata.getvalue()
    return html