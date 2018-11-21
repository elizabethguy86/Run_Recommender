# Run_Recommender
A Run Recommendation Program Utilizing the Strava API

### Goal - to produce readable maps of suggested running routes based on user input of preferred distance, location, and elevation gain.

**Overview:** 
[Strava](https://www.strava.com) is a popular web and mobile application used by runners, bikers, and other athletes to track and share athletic activities. In the running community, a benefit of sharing activities is that other approved users (known as “connections”) can view shared activities, and be inspired to try out new running routes. However, searching for routes requires tedious sifting through the running history of connections. Furthermore, there is not a streamlined method of searching for popular running routes using parameters such as location, distance, and elevation gain.  

The proposed solution to the route finding issue is to utilze runs collected from other Strava members and recommend runs based on similarity to a list of preferred location proximity, distance, and elevation gain.  This application utilizes the Strava API to collect runs from approved users through this [link](https://route.dsi.link/)(see also [this_github](https://github.com/elizabethguy86/runpaw)).
