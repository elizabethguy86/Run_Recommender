# Run_Recommender
A Run Recommendation Program Utilizing the Strava API

### Goal - to produce readable maps of suggested running routes based on user input of preferred distance, location, and elevation gain.

### Table of Contents:

1. [Overview](#overview)
2. [Model](#paragraph1)
3. [Website](#paragraph2)
3. [Instructions](#paragraph3)

**Overview:** <a name="overview"></a>
[Strava](https://www.strava.com) is a popular web and mobile application used by runners, bikers, and other athletes to track and share athletic activities. In the running community, a benefit of sharing activities is that other approved users (known as “connections”) can view shared activities, and be inspired to try out new running routes. However, searching for routes requires tedious sifting through the running history of connections. Furthermore, there is not a streamlined method of searching for popular running routes using parameters such as location, distance, and elevation gain.  

This solution to the route finding issue utilizes runs collected from other Strava members, and recommends runs based on similarity to a list of preferred location proximity, distance, and elevation gain.  Users approve access to their Strava data through the [website](http://pnwrun.org/).

**Model:** <a name="paragraph1"></a>
The model takes in running routes from approved Strava users.  The json data is converted into a dictionary and read into a pandas dataframe with the following features:  'upload_id','average_speed', 'distance', 'elapsed_time', 'total_elevation_gain', 'type', 'start_date_local','start_latlng', 'end_latlng', 'map'.  Only activities of 'type' == 'run' are utilized.  The resultant dataframe is further filtered by examining the great circle distance from the current location to the 'start_latlng'.  If the activity start distance is <= the user-specified distance away from their preferred location, the activity is included. Then, the user specifies their preferred distance to run and elevation gain.  The 20 most similar runs based on euclidean distance are then selected.  Some of those runs are repeats of the same route. In order to return a variety of run routes, runs are grouped by hierarchical clustering according to cosine similarity, using the standardized centroid of the lat, long coordinates for each route (converted to cartesian coordinate space) and elevation gain.  Then, 5 routes are selected from the resultant groups.  Groups with large numbers of runs are prioritized for selection, as runs that are often repeated are presumably "popular" routes.

**Tools Used:**
* Data Storage/Manipulation: AWS/MongoDB and Pandas
* Clustering: Scipy and Scikit Learn
* Map Display: Leaflet
* Website: Flask and AWS (web hosting)

![tools](https://github.com/elizabethguy86/Run_Recommender/blob/master/Presentation/Tools_Used.jpg)

**Model Measurement:**
User feedback was collected via survey with a 5 point Likert scale ([survey link](https://goo.gl/forms/UMuNlv6t8kjVNb1q1)).  Features measured were run variety, specificity of run metrics, and likelihood of running a suggestd route.

**Website:**<a name="paragraph2"></a>
Try out the Run_Recommender by visiting [pnwrun.org](http://pnwrun.org/). The `index.html` file in templates works in conjunction with the `app.py` file.  `app.py` can be run on a FLASK server using the command `FLASK_APP=app.py flask run`.

**Instructions:**<a name="paragraph3"></a>
The `model` folder contains functions and classes for setting up the activities dataframe, filtering data, and grouping routes that are copies of each other in order to return unique routes to the users.  Use `Setup_Initial_DataFrame.py` as a template to create your initial dataframe of activities. The template uses an instance of the Activities class to create a dataframe containing features specified in the cols variable.  Note that user tokens (required for data access) are stored in a MongoDB on AWS in this example.  Thus, code to extract tokens will differ from situation to situation.

Once your dataframe is created, you can use the `Run_Recommender()` class from
`Filter_Data.py`.  Run_Recommender is initialized with the dataframe and a (lat, long) tuple for starting coordinates.  Recommend runs by calling self.recommend_runs(), which requires a request input: list of [elevation gain(meters), distance(miles) to run]) and a distance(float) in miles that you are willing to travel from the starting coordinates for a run.

Several of the runs in the database are repeats of the same route, or very similar to each other.  To group similar runs together, use the `Group_Runs.py` file and the `GroupRuns()` class.  The `map_coordinates` method will return a list of all the map coordinates from your polylines, retrieved from the `Run_Recommender()` class.  The `make_groups` method will group similar runs together, and then select one run from each group (prioritizing selecting from large groups first, as frequently-run routes are likely more popular).  `make_groups` returns indices that can be used to filter through the list of map coordinates from the `map_coordinates` method for later mapping of the suggested routes.

Use the functions in `Mapping_Functions` to make maps of your suggested routes.  Use of the classes to retrieve unique routes and route coordinates are provided as commented-out examples.
