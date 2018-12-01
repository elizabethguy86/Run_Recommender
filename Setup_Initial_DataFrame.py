import pandas as pd
import numpy as np
from Activities_Dictionary import Activities
from stravalib.client import Client
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale
from math import radians, cos, sin, asin, sqrt

'''Get tokens from MongoDB'''
tmux
db.strava_tokens.find()

'''Remap localhost to EC2 server in bash'''
! ssh -NL 47017:localhost:27017 quesadilla

'''Get list of users and their tokens from MongoDB.
Instructions: Set up tunnel ssh ssh -NL 47017:localhost:27017 #your EC2 name'''
mc = MongoClient(host='localhost:47017')
users = mc['runpaw']
tokens = list(users.strava_tokens.find())

#columns to use:
cols = ['upload_id',
        'average_speed',
        'distance',
        'elapsed_time',
        'total_elevation_gain',
        'type',
        'start_date_local',
        'start_latlng',
        'end_latlng', 'map']

Activities = Activities(tokens, cols)
df = Activities.activities_to_dict()

'''Only use runs'''
df = df[df['type'] == 'Run']