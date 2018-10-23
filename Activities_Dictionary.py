from stravalib.client import Client
from stravalib import unithelper
import pandas as pd

class Activities:
    '''Takes in activities JSON object and converts it into
    a dictionary object. Converts dictionary object into
    pandas dataframe.'''

    def __init__(self, activites):
        #activities from client.get_activities()
        self.activities = activities

    def make_dataframe(self, columns):
        '''take activities dictionary and convert it to dataframe'''
        activities = self.activities
        cols = columns #columns should be in list
        data = []
        for activity in activities:
            activity_dict = activity.to_dict()
            data.append([activity_dict.get(x) for x in cols])
        return pd.DataFrame(data, columns=cols)
