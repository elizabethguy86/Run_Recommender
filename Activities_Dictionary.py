from stravalib.client import Client
import pandas as pd

class Activities:
    '''Takes in activities JSON object and converts it into
    a dictionary object. Converts dictionary object into
    pandas dataframe.'''

    def __init__(self, tokens, cols):
        #activities from client.get_activities()
        self.tokens = tokens 
        self.cols = cols

    def activities_to_dict(self):
    data = []
    tokens = self.tokens
    for token in tokens:
    client_current = Client(access_token=token)
    activities = client_current.get_activities(limit=1000)
    for activity in activities:
        my_dict = activity.to_dict()
        data.append([my_dict.get(x) for x in self.cols])
    #make large dataframe for columns of interest for all tokens
    df = pd.DataFrame(data, columns=self.cols)
    #convert distance to miles
    df['miles_converted'] = [x/1609.3440122044242 for x in df['distance']]
    return df
