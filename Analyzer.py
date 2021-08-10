from pymongo import MongoClient
import numpy as np
import pandas as pd
import multiprocessing
import pickle
from collections import Counter
from pprint import pprint
from IPython.display import display
##
import nltk
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
##
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
##
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class Analyzer:
    '''This class is used to analyze reviews'''
    def __init__(self,path=None):
        if path != None:
            self.model = Model.load(path,pickle.load)
        else:
            #get from data pipeline
            pass

    def connect_mongo(self):
        '''connect to mongo db and create the required collections'''
        client = MongoClient('localhost',27017)
        db = client['Steam']
        self.reviews = db['Reviews'] # appid {review}
        self.apps = db['Apps']
        self.ids = self.reviews.find().distinct('appid')
    def get_all_game_reviews(self):
        '''Get all game reviews from mongo'''
        r = self.reviews.find({},{"reviews.review":1,"reviews.voted_up":1, "_id":0})
        lst = []
        for reviews in list(r):
            for review in reviews['reviews']:
                lst.append((review['review'],review['voted_up']))
        return lst
    def get_game_reviews(self, appid):
        '''
        Get game reviews by game id
        '''
        r = self.reviews.find({'appid':appid},{"reviews.review":1,"reviews.voted_up":1, "_id":0})
        lst = []
        for reviews in list(r):
            for review in reviews['reviews']:
                lst.append((review['review'],review['voted_up']))
        print(len(lst))
        return lst

class Grahpics:
    '''This class is for generating visuals with the data'''
    def __init__(self):
        ''''''
        pass
    def word_cloud(self,words,stopwords,path=None):
        '''Create a word cloud'''
        wordcloud = WordCloud(width = 1600, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate_from_frequencies(words.T.sum(axis=1))
 
        # plot the WordCloud image                      
        fig = plt.figure( facecolor = None)
        fig.set_size_inches(90,45)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        if path != None:
            self.save(path,fig)
        else:
            plt.show()

    def graph(self):
        '''Create a graph'''
        pass
    def save(self,path,fig):
        '''Save the graphic passed'''
        fig.savefig(path, dpi=10)


class DataPipeline:
    '''This class handles loading and cleaning data'''
    def __init__(self, reviews, apps, ids):
        ''''''
        pass
    def load_data(self):
        '''Load data from specified location'''
        pass
    def preprocess(self,X,y=None):
        '''Steps before main processing'''
        print('pre-process')
        
        return X,y

    def process(self, X, y=None):
        '''
        Main processing of the data for training and testing
        X data in Returns data in proper formatting
        '''
        X,y = self.preprocess(X,y)
        print('process')
        X,y = self.postprocess(X,y)
        return X,y

    def postprocess(self,X,y=None):
        '''Steps After main processing'''
        print('post-process')
        return X,y


    
    #spark? mongo?
    @staticmethod
    def create_tfidf_dictionary(x, transformed_file, features):
        '''
        create dictionary for each input sentence x, where each word has assigned its tfidf score
        '''
        vector_coo = transformed_file[x.name].tocoo()
        vector_coo.col = features.iloc[vector_coo.col].values
        dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
        return dict_from_coo
    @staticmethod
    def replace_tfidf_words(x, transformed_file, features):
        '''
        replacing each word with it's calculated tfidf dictionary with scores of each word
        '''
        dictionary = DataPipeline.create_tfidf_dictionary(x, transformed_file, features)   
        # pprint(dictionary)
        # pprint(x.corpus.split())
        return list(map(lambda y:dictionary[y], x.corpus.split()))
    @staticmethod
    def replace_sentiment_words(word, sentiment_dict):
        '''
        replacing each word with its associated sentiment score from sentiment dict
        '''
        try: out = sentiment_dict[word]
        except KeyError: out = 0
        return out
    @staticmethod
    def preprocess_train(data):
        '''
        take list data and return X,y
        '''
        if DataPipeline.data_has_Y(data):
            voted = [b for _,b in data]
            voted = [1 if x == True else 0 for x in voted]

            corpus = [a for a,_ in data]
            corpus = [c.lower() for c in corpus]
            return corpus, voted
        else :
            corpus = [a for a,_ in data]
            corpus = [c.lower() for c in corpus]
            return corpus, None
    @staticmethod
    def data_has_Y(X):
        for x in X:
            if isinstance(x,tuple):
                return True
            else:
                return False
    @staticmethod
    def clean(text):
        wn = nltk.WordNetLemmatizer()
        stopword = nltk.corpus.stopwords.words('english')
        tokens = nltk.word_tokenize(text)
        lower = [word.lower() for word in tokens]
        no_stopwords = [word for word in lower if word not in stopword]
        no_alpha = [word for word in no_stopwords if word.isalpha()]
        lemm_text = [wn.lemmatize(word) for word in no_alpha]
        clean_text = lemm_text
        return clean_text
    @staticmethod
    def vectorize(data,tfidf_vect_fit):
        X_tfidf = tfidf_vect_fit.transform(data)
        words = tfidf_vect_fit.get_feature_names()
        X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
        X_tfidf_df.columns = words
        return(X_tfidf_df)
    # @staticmethod
    # @staticmethod
    # @staticmethod
    # @staticmethod
    # @staticmethod

class Model:
    ''''''
    def __init__(self,build_model,score_func,save_func):
        '''
        Parameters:

        build_model:   function taht will build the model
        ex. build_model = Model.xgb || Model.randomForest || cluster || LSTM?
        '''
        self.build_model_call = build_model
        self.score_func = score_func
        self.save_func = save_func
    def build_model(self):
        '''The function that determines what self.model will be'''
        print('build model start')
        self.model = self.build_model_call()
        print('build model end')
    def fit(self,X,y=None):
        '''Fits the model with provided X,y data'''
        self.model.fit(X,y)
    def search(self):
        '''Use this to grid search'''
        print('temporarily inactive')
    def predict(self,X):
        '''Make a prediction on given X data'''
        return self.model.predict(X)
    def score(self,y_true,y_hat):
        '''Evaluate a set of predictions'''
        self.score_func(y_true,y_hat)
    def predict_and_score(self,X,y_true):
        '''Evaluate a set of predictions in one function'''
        y_hat=self.predict(X)
        print(len(y_true))
        self.score_func(y_hat,y_true)
    def save(self):
        '''Save the current model'''
        self.save_func(self)

    ## Build models
    @staticmethod
    def make_random_forest():
        rf = RandomForestClassifier(n_jobs=-1)
        return rf
    @staticmethod
    def make_xgb_forest():
        xgb = GradientBoostingClassifier()
        return xgb
    @staticmethod
    def cluster():
        km = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50)
        return km
    @staticmethod
    def make_tfidf_model(temp):
        tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
        tfidf.fit(temp)
        
        features = pd.Series(tfidf.get_feature_names())
        transformed = tfidf.transform(temp)
        return transformed,features
    @staticmethod
    def make_w2v_model(sentences):
        w2v_model = Word2Vec(min_count=1,
                            window=4,
                            vector_size=300,
                            sample=1e-5, 
                            alpha=0.03, 
                            min_alpha=0.0007, 
                            negative=20,
                            workers=multiprocessing.cpu_count()-1)

        w2v_model.wv
        w2v_model.build_vocab(sentences, progress_per=10000)
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
        model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=w2v_model.wv.vectors)
        # print(w2v_model.wv.similar_by_vector(negative_cluster_center, topn=10, restrict_vocab=None))
        word_vectors = w2v_model.wv
        words = pd.DataFrame(word_vectors.index_to_key)
        words.columns = ['words']
        words['vectors'] = words['words'].apply(lambda x: word_vectors[f'{x}'])
        words['cluster'] = words['vectors'].apply(lambda x: model.predict(np.array(x).reshape(1, -1)) )
        words['cluster'] = words['cluster'].apply(lambda x: x[0])
        words['cluster_value'] = [1 if i==0 else -1 for i in words.cluster]
        words['closeness_score'] = words.apply(lambda x: 1/(1-model.transform([x.vectors]).min()), axis=1)
        words['sentiment_coeff'] = words['closeness_score'] * words.cluster_value
        sentiment_dict = dict(zip(words['words'].values, words['sentiment_coeff'].values))
        return sentiment_dict
    ##
    @staticmethod
    def pickle_load(path):
        print('load')
        with open(path+'.pickle', 'rb') as handle:
            b = pickle.load(handle)
        return b
    @staticmethod
    def pickle_save(path,data):
        print('save')
        with open(path+'.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved')
    @staticmethod
    def temp_score(y_test, predicted_classes,label="model"):
        print(len(y_test),'y test')
        print(len(predicted_classes),'y_pred')
        test_scores = [accuracy_score(y_test,predicted_classes), precision_score(y_test, predicted_classes), recall_score(y_test, predicted_classes), f1_score(y_test, predicted_classes)]

        print(f'\n \n Scores {label}')
        scores = pd.DataFrame(data=[test_scores])
        scores.columns = ['accuracy', 'precision', 'recall', 'f1']
        scores = scores.T
        scores.columns = ['scores']
        display(scores)
    @staticmethod
    def score(replacement_df, predicted_classes, y_test):
        conf_matrix = pd.DataFrame(confusion_matrix(replacement_df.sentiment, replacement_df.prediction))
        print('Confusion Matrix')
        display(conf_matrix)

        test_scores = accuracy_score(y_test,predicted_classes), precision_score(y_test, predicted_classes), recall_score(y_test, predicted_classes), f1_score(y_test, predicted_classes)

        print('\n \n Scores')
        scores = pd.DataFrame(data=[test_scores])
        scores.columns = ['accuracy', 'precision', 'recall', 'f1']
        scores = scores.T
        scores.columns = ['scores']
        display(scores)
        



#arg inputs
    #run it with spark?

#refit api to use analyzer
#add mode app routes

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

if __name__ == '__main__':
    print('++++++++++++')

    stopwords = nltk.corpus.stopwords.words('english')
    a = Analyzer()
    a.connect_mongo()

    reviews = a.get_all_game_reviews()
    
    #
    revs = a.get_game_reviews(a.ids[0])
    b,_ = DataPipeline.preprocess_train(revs)
    tfidf_vect = TfidfVectorizer(analyzer=DataPipeline.clean)
    print('fitting vec')
    tfidf_vect_fit=tfidf_vect.fit(b)
    print('vectorizing X')
    X_train=DataPipeline.vectorize(b,tfidf_vect_fit)
    cloud = Grahpics()
    print('heyo')
    cloud.word_cloud(X_train,stopwords,'./vue/client/public/images/cloud3.png')
    #

    pipe = DataPipeline(a.reviews,a.apps,a.ids)
    X,y = DataPipeline.preprocess_train(reviews)

    X,y = pipe.process(X,y)
    print(len(X),len(y))
    
    temp_y = []
    temp_X = []
    for index,x in enumerate(X):
        if x not in temp_X:
            temp_X.append(x)
        else:
            temp_y.append(index)
    delete_multiple_element(X,temp_y)
    delete_multiple_element(y,temp_y)

    print(len(X),len(y))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.90, random_state=42)
    
    tfidf_vect = TfidfVectorizer(analyzer=DataPipeline.clean)
    print('fitting vec')
    tfidf_vect_fit=tfidf_vect.fit(X_train)
    # tfidf_vect_fit
    print('vectorizing X')
    X_train=DataPipeline.vectorize(X_train,tfidf_vect_fit)
    X_test=DataPipeline.vectorize(X_test,tfidf_vect_fit)
    # pprint(X_train)
    cloud = Grahpics()
    cloud.word_cloud(X_train,stopwords,'./vue/client/public/images/cloud2.png')

    model = Model(Model.make_random_forest,Model.temp_score,Model.pickle_save)
    model.build_model()
    model.fit(X_train,y_train)
    # pprint(X_test)
    model.predict_and_score(X_test,y_test)
    # model.save()
