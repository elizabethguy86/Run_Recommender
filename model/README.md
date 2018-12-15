###Get tokens from MongoDB
This system assumes that tokens are stored in MongoDB on a cloud server.  To get access
to the database, remap localhost to EC2 server in bash:

`! ssh -NL 47017:localhost:27017 [your EC2 name]`

Then create a client:

mc = MongoClient(host='localhost:47017')
users = mc['your_database']
tokens = list(users.strava_tokens.find())

Once tokens are acquired, the run activities can be retrieved from Strava and stored in a dataframe as demonstrated in `Setup_Initial_Dataframe.py`.