import requests
from pymongo import MongoClient
import pickle
from pprint import pprint
class API:
    def __init__(self):
        self.connect_to_mongo()
        # self.load_model()
        
    def connect_to_mongo(self):
        client = MongoClient('localhost',27017)
        db = client['Steam']
        self.reviews = db['Reviews'] # appid {review}
        self.apps = db['Apps']
    
    def load_model(self):
        self.model = pickle.load('model.pickle')

    def predict(self,text):
        return 'This review will be liked' #self.model.predict(text)

    def request_app_reviews(self):#filter
        pass
    def request_all_reviews(self):
        lst = [x for x in self.reviews.find({"$where":"this.query_summary.num_reviews > 1"},{"reviews":1,'query_summary':1,"_id":0,"appid":1}).limit(100)]
        temp = lst[0]
        new_list = [self.clean_review(r,temp['appid']) for r in temp['reviews'] ]
        return new_list
        
    def get_name_from_id(self,id):
        return list(self.apps.find({"appid":id},{"name":1,"_id":0}))[0]['name']
    def return_app_reviews(self):
        pass
    def return_random_review(self): #sample
        pass

    def return_app_list(self):
        lst = list(self.apps.find().limit(100))
        return [self.clean_app(x) for x in lst]
    
    def clean_review(self,rev,appid):
        name = self.get_name_from_id(appid)
        myreview = {
            'name': name,
            'text': rev['review'],
            'steamid' : rev['author']['steamid'],
            'num_games_owned' : rev['author']['num_games_owned'],
            'num_reviews' : rev['author']['num_reviews'],
            'playtime_forever' : rev['author']['playtime_forever'],
        }
        return myreview
        
    def clean_app(self, app):
        price = int(app['price'])/100
        game = {
            'name': app['name'],
            'developer': app['developer'],
            'publisher': app['publisher'],
            'price': price if price > 0 else 'Free', #get as dollar value
            'owners': str(app['owners']).replace('..','-'),
        }
        return game

    def dashReviews(self):
        rev = 0#len(list(self.reviews.find({}).distinct('appid')))
        auth = 0#len(list(self.reviews.find({}).distinct('developer')))
        return {
            'reviewCount':rev,
            'authorCount':auth,
            'avgPerAuth':0,
            'avgPerGame':0,
        }
    def dashGames(self): #sample
        games = len(list(self.apps.find({}).distinct('appid')))
        devs = len(list(self.apps.find({}).distinct('developer')))
        pubs = len(list(self.apps.find({}).distinct('publisher')))

        # avgs = list(self.apps.find({},{'price':1}))
        # print(type(avgs[0]['price']), 'stop me')

        # avg = list(self.apps.aggregate(
        #     [
        #         {
        #         '$price':
        #             {  
        #             'avg': {'price': { '$toInt': '$price' } }
        #             }   
        #         }
        #     ]))
        # print(avg)
        # avg = int(avg)/100
        return {
            'gameCount':games,
            'devCount':devs,
            'pubCount':pubs,
            'avgPrice':10
        }


if __name__ == '__main__':
    api = API()
    # game = api.clean_app(api.return_app_list()[0])
    # pprint(game)


