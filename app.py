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

@app.route('/') #landing page
def hello_world():
    return render_template('index.html', table=table_practice()[0], map='<iframe src="/map" width="1000" height="500"> </iframe>')

#@app.route('/request', methods=['POST'])
def make_recommendations():
    recommendations = Run_Recommender(df, (data['user_input_location']))
    recommend_dict, similarities = recommendations.recommend_runs([data['user_input']], 5)
    polylines, indices = recommendations.make_polyline_dict()
    Group = GroupRuns(polylines, indices, df)
    map_coordinates = Group.map_coordinates()
    indices_to_use = Group.make_groups(threshold=0.05)
    unique_coordinates = [map_coordinates[i] for i in indices_to_use]
    mapping_dict = mapfun.map_indices(indices_to_use, indices)
    stats = return_route_stats(mapping_dict, indices_to_use, df)
    abbrev_stats = stats.loc[:, ['total_elevation_gain', 'miles_converted']]
    return abbrev_stats.to_html()

def table_practice():
    recommendations = Run_Recommender(df, (47.508802, -122.464284))
    recommend_dict, similarities = recommendations.recommend_runs([200,15], 5)
    polylines, indices = recommendations.make_polyline_dict()
    Group = GroupRuns(polylines, indices, df)
    map_coordinates = Group.map_coordinates()
    indices_to_use = Group.make_groups(threshold=0.05)
    unique_coordinates = [map_coordinates[i] for i in indices_to_use]
    mapping_dict = mapfun.map_indices(indices_to_use, indices)
    stats = mapfun.return_route_stats(mapping_dict, indices_to_use, df)
    abbrev_stats = stats.loc[:, ['total_elevation_gain', 'miles_converted']]
    return abbrev_stats.to_html(), unique_coordinates


@app.route('/map')
def map_runs():
    unique_coordinates = table_practice()[1]
    #get start point for the map
    lat, long = unique_coordinates[0][0]
    m = folium.Map(location=[lat, long], zoom_start=12.2)
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
        &nbsp; Route_0 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:blue"></i><br>
        &nbsp; Route_1 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:green"></i>
        &nbsp; Route_2 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:red"></i><br>
        &nbsp; Route_3 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:orange"></i>
        &nbsp; Route_4 &nbsp; <i class="fa fa-square fa-2x"
                    style="color:purple"></i>
        </div>'''
    m.get_root().html.add_child(folium.Element(legend_html)) #add legend to map
    #html_string = m.get_root().render() #get map html
    #m.save('test.html')
    mapdata = BytesIO()
    m.save(mapdata, close_file=False)
    html = mapdata.getvalue()
    return html