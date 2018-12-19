import pandas as pd
import numpy as np
from Activities_Dictionary import Activities
from pymongo import MongoClient


'''Get list of users and their tokens from MongoDB.
Instructions: Set up tunnel ssh ssh -NL 47017:localhost:27017 #your EC2 name'''
mc = MongoClient(host='localhost:47017')
users = mc['runpaw']
tokens = list(users.strava_tokens.find())

def make_tokens(tokens):
    '''Take a list of users from MongoDB and extract their
       tokens from the dictionary objects in the list'''
    tokens_set = set() #create set to avoid multiple copies of token
    for token in tokens:
        tokens_set.add(token['token'])
    return tokens_set

#get all the unique tokens
tokens_unique = make_tokens(tokens)

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

events = Activities(tokens, cols)
df = events.activities_to_dict()

'''Only use runs'''
df = df[df['type'] == 'Run']
