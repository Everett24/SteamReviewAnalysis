# from flask import Flask, render_template, Blueprint
from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
from API import API
from pprint import pprint
# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

api = API()

@app.route('/predict', methods=['POST'])
def predict():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        post_data = request.get_json()
        text = post_data.get('text')
        res =  api.predict(text)
        response_object['message'] = res
   
    return jsonify(response_object)

@app.route('/Games', methods=['GET'])
def get_games():
    response_object = {'status': 'success'}
    response_object['games'] = api.return_app_list()
    return jsonify(response_object)

@app.route('/Reviews', methods=['GET'])
def get_reviews():
    response_object = {'status': 'success'}
    response_object['reviews'] = api.request_all_reviews()
    # pprint(response_object)
    return jsonify(response_object)

@app.route('/dashGames', methods=['GET'])
def get_dash_games():
    response_object = {'status': 'success'}
    response_object['games'] = api.dashGames()
    return jsonify(response_object)

@app.route('/dashReviews', methods=['GET'])
def get_dash_reviews():
    response_object = {'status': 'success'}
    response_object['reviews'] = api.dashReviews()
    return jsonify(response_object)


if __name__ == '__main__':
    app.run()
