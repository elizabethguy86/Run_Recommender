# Run_Recommender
A Run Recommendation Program Utilizing the Strava API

### Goal - to produce readable maps of suggested running routes based on user input of preferred distance, location, and elevation gain.

**Overview:** 
[Strava](https://www.strava.com) is a popular web and mobile application used by runners, bikers, and other athletes to track and share athletic activities. In the running community, a benefit of sharing activities is that other approved users (known as “connections”) can view shared activities, and be inspired to try out new running routes. However, searching for routes requires tedious sifting through the running history of connections. Furthermore, there is not a streamlined method of searching for popular running routes using parameters such as location, distance, and elevation gain.  

The proposed solution to the route finding issue is to utilze runs collected from other Strava members and recommend runs based on similarity to a list of preferred location proximity, distance, and elevation gain.  This application utilizes the Strava API to collect runs, approved by users through this [link](https://route.dsi.link/)(see also [this_github](https://github.com/elizabethguy86/runpaw)).

**Instructions:**
Use `Setup_Initial_DataFrame.py` as a template to create your initial dataframe of activities. Uses instance of the Activities class to create a dataframe containing features specified in the cols variable.  Note that user tokens (required for data access) are stored in a MongoDB on AWS in this instance.  Thus, code to extract tokens will differ from situation to situation.

Once your dataframe is created, you can use the `Run_Recommender()` class from 
`Filter_Data.py`.  Run_Recommender is initialized with the dataframe and a (lat, long) tuple for starting coordinates.  Recommend runs by calling self.recommend_runs(), which requires a request input, list of [elevation gain(meters), distance(miles) to run]) and a distance in miles that you are willing to travel from the starting coordinates for a run.

Several of the runs in the database are repeats of the same route, or very similar to each other.  To group similar runs together, use the `Group_Runs.py` file and the `GroupRuns()` class.  The `map_coordinates` method will return a list of all the map coordinates from your polylines, retrieved from the `Run_Recommender()` class.  The `make_groups` method will group similar runs together, and then select one run from each group (prioritizing selecting from large groups first, as frequently-run routes are likely more popular).  `make_groups` returns indices that can be used to filter through the list of map coordinates from the `map_coordinates` method for later mapping of the suggested routes.