import requests
import pandas as pd
from pymongo import MongoClient
from pprint import pprint

class Acquisitions:
    '''This class is to gather reviews and apps from the steam api'''
    def __init__(self):
        self.connect_mongo()

    def connect_mongo(self):
        '''connect to mongo db and create the required collections'''
        client = MongoClient('localhost',27017)
        db = client['Steam']
        self.reviews = db['Reviews'] # appid {review}
        self.apps = db['Apps']
        self.ids = self.reviews.find().distinct('appid')
        
    def run(self):
        '''run the aquistions object, request and save reviews'''
        self.request()
        #take in arguments? this is a console object? if main?

    def request(self):
        '''request reviews from steam api'''
        ids = self.reviews.find().distinct('appid')
        skip  = len(list(self.apps.find({"appid":{"$in":ids}}))) ### this should change

        for i,document in enumerate(self.apps.find().skip(skip+2)):
            appid = document['appid']
            self.save(appid,i) 
    
    def save(self,appid,i=1,cursor='*'):
        '''save reviews to mongo collection'''
        try:
            temp = requests.get(f'http://store.steampowered.com/appreviews/{appid}?json=1&num_per_page=1&cursor=*')
            print(temp)
        except:
            return
        reviews_count = temp.json()['query_summary']['total_reviews']
        cnt = reviews_count if reviews_count < 10000 else 10000 

        for x in range(cnt):
            s = f'http://store.steampowered.com/appreviews/{appid}?json=1&num_per_page=100&cursor={cursor}'
            res = requests.get(s)
            j = res.json()
            # pprint(j)
            if 'cursor' in j.keys():
                cursor = j['cursor'].encode()
            else:
                break
            j['appid'] = appid
            self.reviews.insert_one(j)



if __name__ == '__main__':
    print('++++++++++++')
    acq = Acquisitions()
    acq.run()#add offset bool # pre save duplicate check?


    def inspect():
        pass
    def sample():
        pass
    # inspect()
    # sample()
