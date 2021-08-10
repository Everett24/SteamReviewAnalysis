import requests
import pandas as pd
from pymongo import MongoClient
from pprint import pprint

client = MongoClient('localhost',27017)
db = client['Steam']
reviews = db['Reviews'] # appid {review}
apps = db['Apps']
ids = reviews.find().distinct('appid')

print('unique apps: ',apps.count_documents({}))
print('Reviews Count ',reviews.count_documents({}))

def make_requests():
    
    cnt = apps.count_documents({})
    ids = reviews.find().distinct('appid')
    skip  = len(list(apps.find({"appid":{"$in":ids}}))) ### this should change

    for i,document in enumerate(apps.find().skip(skip+2)):
        appid = document['appid']
        save_reviews(appid,i) 
        r_cnt = apps.count_documents({'appid':document['appid']})


def save_reviews(appid,i=1,cursor='*'):
    # appid = 578080
    # cursor = '*'
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
        reviews.insert_one(j)


# save_reviews(578080)

# make_requests()
# rapps = list(apps.find({"appid":{"$in":ids}}))
# print('Reviewed games: ', len(rapps))
# for app in rapps:
#     print(app['name'])

# for app in apps.find({}).limit(36): #.aggregate([{ "$sample": { "size": 20 } }]): ############sample
#     print(app['name'])


# print(reviews.find_one({}))

# print(reviews.find_one({})['reviews'][0].keys()) #timestamp , votes up/down, comments, purchase

# print(reviews.find_one({})['reviews'][0]['author']) # author object
# print(reviews.find_one({})['reviews'][0]['review']) # the raw text

# a = reviews.find_one({})['reviews'][0]['author']
# print(type(a))
# df = pd.json_normalize(a)
# print(df.head())

# b = reviews.find_one({})['reviews']
# print(type(b))
# df2 = pd.json_normalize(b)
# print(df2.head())
# print(df2.columns)
# print(df2['review'])

if __name__ == '__main__':
    print('')