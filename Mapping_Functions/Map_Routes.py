
from Activities_Dictionary import Activities
from Filter_Data import Run_Recommender
from Group_Runs import GroupRuns
import folium

'''initialize the map. map_coordinates should come from GroupRuns() class, which 
takes in polylines, indices from your make_polyline_dict() method from Run_Recommender
class'''

#recommendations = Run_Recommender(df, (latitude, longitude))
#recommend_dict = recommendations.recommend_runs([elevation, distance], miles_away) #must do this before creating polylines

# polylines, indices = recommendations.make_polyline_dict()

# Group = GroupRuns(polylines, indices, df)
# map_coordinates = Group.map_coordinates()
# indices_to_use = Group.make_groups(threshold=0.05) """threshold may be changed for cosine sim distance groups"""

def map_indices(indices_to_use, indices):
    '''Takes in indices_to_use from 20 suggested routes and the actual indices of the 20 routes in the larger
    dataframe and returns a mapping of indices_to_use back to the index in the larger dataframe. 
    Use: to retrieve stats for suggested routes'''
    mapping = {}
    for idx, i in enumerate(indices):
        if idx in indices_to_use:
            mapping[idx] = i
    return mapping
        
#mapping_dict = map_indices(indices_to_use, indices)

def return_route_stats(mapping_dict, indices_to_use, df):
    '''Returns the elevation gain and miles for the routes that will be returned'''
    mapping = [mapping_dict[i] for i in indices_to_use[0:5]]
    slice_df = df.iloc[mapping]
    return slice_df.loc[:, ['total_elevation_gain', 'miles_converted']].reset_index()

#dataframe of indices from original df, elevation gain(m), and the miles of each route
#return_route_stats(mapping_dict, indices_to_use, df)

#get start point for the map
# lat, long = map_coordinates[0][0]
# m = folium.Map(location=[lat, long], zoom_start=12.2)

# for idx, route in enumerate(unique_coordinates[0:5]):
#     colors = ['blue','green','red','orange','purple']
#     folium.PolyLine(
#             route,
#             weight=2,
#             color=colors[idx]
#         ).add_to(m)

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
    
#m.get_root().html.add_child(folium.Element(legend_html)) #add legend to map
#m