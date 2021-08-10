import numpy as np
import pandas as pd
import multiprocessing
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


stopwords = nltk.corpus.stopwords.words('english')
##db
client = MongoClient('localhost',27017)
db = client['Steam']
reviews = db['Reviews'] 
apps = db['Apps']
ids = reviews.find().distinct('appid') #get all unique game ids


def create_tfidf_dictionary(x, transformed_file, features):
    '''
    create dictionary for each input sentence x, where each word has assigned its tfidf score
    '''
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo

def replace_tfidf_words(x, transformed_file, features):
    '''
    replacing each word with it's calculated tfidf dictionary with scores of each word
    '''
    dictionary = create_tfidf_dictionary(x, transformed_file, features)   
    # pprint(dictionary)
    # pprint(x.corpus.split())
    return list(map(lambda y:dictionary[y], x.corpus.split()))

def get_game_reviews(appid,reviews):
    '''
    
    '''
    r = reviews.find({'appid':appid},{"reviews.review":1,"reviews.voted_up":1, "_id":0})
    lst = []
    for reviews in list(r):
        for review in reviews['reviews']:
            lst.append((review['review'],review['voted_up']))
    return lst

def get_all_game_reviews(reviews):
    r = reviews.find({},{"reviews.review":1,"reviews.voted_up":1, "_id":0}).limit(25000)
    lst = []
    for reviews in list(r):
        for review in reviews['reviews']:
            lst.append((review['review'],review['voted_up']))
    # print(len(lst))
    return lst

def replace_sentiment_words(word, sentiment_dict):
    '''
    replacing each word with its associated sentiment score from sentiment dict
    '''
    try: out = sentiment_dict[word]
    except KeyError: out = 0
    return out

def process(corpus):
    '''
    
    '''
    file_weighting,temp,sentences = preprocess(corpus)

    transformed,features = make_tfidf_model(temp)
    replacement_df, predicted_classes,y_test = post_process(file_weighting, transformed, features, sentiment_dict=make_w2v_model(sentences))
    score(replacement_df, predicted_classes,y_test)
    

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

def make_tfidf_model(temp):
    tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
    tfidf.fit(temp)
    
    features = pd.Series(tfidf.get_feature_names())
    transformed = tfidf.transform(temp)
    return transformed,features

def tokenize_corpus():
    pass

def cluster(X):
    km = KMeans()
    km.fit(X)
    return km

def make_random_forest(X,y):
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(X,y)
    
    print(rf.feature_importances_)
    return rf

def make_xgb_forest(X,y):
    xgb = GradientBoostingClassifier()
    xgb.fit(X,y)
    # print(xgb.feature_importances_)
    return xgb

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

def run():
    pass

def preprocess(corpus):
    voted = [b for _,b in corpus]
    voted = [1 if x == True else 0 for x in voted]
    corpus = [a for a,_ in corpus]
    corpus = [c.lower() for c in corpus]
    temp = corpus

    stop_words = set(stopwords.words('english'))
    word_tokens = []
    for c in corpus:
        word_tokens.extend(word_tokenize(c))

    corpus = [w for w in word_tokens if (not w in stop_words) & (w.isalpha())]
    words = [x.split() for x in corpus]
    phrases = Phrases(words , min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[words]
   
    file_weighting = pd.DataFrame()
    file_weighting['corpus'] = temp
    file_weighting['voted'] = voted
    return file_weighting,temp,sentences

def post_process(file_weighting, transformed, features, sentiment_dict):
    replaced_tfidf_scores = file_weighting.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)
    replaced_closeness_scores = file_weighting.corpus.apply(lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))

    replacement_df = pd.DataFrame(data=[replaced_closeness_scores, replaced_tfidf_scores, file_weighting.corpus, file_weighting.voted]).T
    replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence', 'sentiment']
    replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
    replacement_df['prediction'] = (replacement_df.sentiment_rate>0).astype('int8')
    replacement_df['sentiment'] = [1 if i==1 else 0 for i in replacement_df.sentiment]

    predicted_classes = replacement_df.prediction
    y_test = replacement_df.sentiment
    return replacement_df, predicted_classes,y_test

def new_process():
    pass

# process(get_game_reviews(ids[1],reviews))

def temp_score(y_test, predicted_classes,label):
    test_scores = accuracy_score(y_test,predicted_classes), precision_score(y_test, predicted_classes), recall_score(y_test, predicted_classes), f1_score(y_test, predicted_classes)

    print(f'\n \n Scores {label}')
    scores = pd.DataFrame(data=[test_scores])
    scores.columns = ['accuracy', 'precision', 'recall', 'f1']
    scores = scores.T
    scores.columns = ['scores']
    display(scores)

def vectorize(data,tfidf_vect_fit):
    X_tfidf = tfidf_vect_fit.transform(data)
    words = tfidf_vect_fit.get_feature_names()
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    X_tfidf_df.columns = words
    return(X_tfidf_df)

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

reviews = get_all_game_reviews(reviews)
voted = [b for _,b in reviews]
voted = [1 if x == True else 0 for x in voted]
corpus = [a for a,_ in reviews]
corpus = [c.lower() for c in corpus]

# wordcloud = WordCloud(width = 800, height = 800,
#                 background_color ='white',
#                 stopwords = stopwords,
#                 min_font_size = 10).generate(corpus[0])
 
# # plot the WordCloud image                      
# fig = plt.figure( facecolor = None)
# fig.set_size_inches(90,45)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 0)
 
# fig.savefig('cloud.png', dpi=10)
# plt.show()

df = pd.DataFrame()
df['corpus'] = corpus
df['voted'] = voted
voted = np.array(voted)
print( len(voted[voted == 0]) , '0')
print( len(voted[voted == 1]) , '1' )

X_train, X_test, y_train, y_test = train_test_split(corpus,voted,test_size = 0.90, random_state=42)
pprint(np.unique(y_test))
tfidf_vect = TfidfVectorizer(analyzer=clean)
tfidf_vect_fit=tfidf_vect.fit(X_train)
X_train=vectorize(X_train,tfidf_vect_fit)
X_test=vectorize(X_test,tfidf_vect_fit)

rf = make_random_forest(X_train,y_train)
xgb = make_xgb_forest(X_train,y_train)
r_pred = rf.predict(X_test)
x_pred = xgb.predict(X_test)

temp_score(r_pred,y_test,'Random Forest')
temp_score(x_pred,y_test,'XGB')
temp_score(y_test,r_pred,'Real Random Forest')
temp_score(y_test,x_pred,'Real XGB')

# process(get_all_game_reviews(reviews))


if __name__ == '__main__':
    print('')