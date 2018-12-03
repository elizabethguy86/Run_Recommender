
from Activities_Dictionary import Activities
import folium

'''initialize the map. map_coordinates should come from your make_polyline_lst function,
which returns a dictionary of polylines'''

recommendations = Run_Recommender(df, (latitude, longitude))
recommend_dict = recommendations.recommend_runs([100, 5], 3) #must do this before creating polylines

polylines, indices = recommendations.make_polyline_dict()
map_coordinates = []
for line in list(polylines.values()):
    coordinates = polyline.decode(line)
    map_coordinates.append(coordinates)

#get start point for the map
lat, long = map_coordinates[0][0]
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
     &nbsp; Route_1 &nbsp; <i class="fa fa-square fa-2x"
                  style="color:blue"></i><br>
     &nbsp; Route_2 &nbsp; <i class="fa fa-square fa-2x"
                  style="color:green"></i>
     &nbsp; Route_3 &nbsp; <i class="fa fa-square fa-2x"
                  style="color:red"></i><br>
     &nbsp; Route_4 &nbsp; <i class="fa fa-square fa-2x"
                  style="color:orange"></i>
     &nbsp; Route_4 &nbsp; <i class="fa fa-square fa-2x"
                  style="color:purple"></i>
    </div>'''
    
m.get_root().html.add_child(folium.Element(legend_html)) #add legend to mapp
m