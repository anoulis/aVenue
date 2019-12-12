from flask import Flask, jsonify, request, json
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import re
import codecs
import time
import ast
import numpy as np
from pprint import pprint
from collections import OrderedDict, defaultdict
import sys
import indexing
#import search
global data_preparation,my_search

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

journals = [
{
'name': 'Journal Name 1',
'title': 'My title 1',
'similarity': 0.15
},
{
'name': 'Journal Name 2',
'title': 'My title 2',
'similarity': 0.13
}
]
final_dict=[]

class Main(Resource):
    def get(self):
        for journal in journals:
            return journal
        return {'journal': None}, 404

    def clear_dict(self):
        final_dict.clear()

    def post(self):
        t1 = time.time()
        parser = reqparse.RequestParser()
        parser.add_argument('title',
            type = str,
            required = True,
            help = "This title field cannot be left blank"
        )
        parser.add_argument('abstract',
            type = str,
            required=True,
            help = "This abstract field cannot be left blank"
        )
        #data = request.get_json();
        data = parser.parse_args()
        searchData = {'title': data['title'], 'abstract': data['abstract']}
        query = searchData['title'] + ' ' + searchData['abstract']
        print(query)
        #search.search_tfidf(query)
        #my_search = search.Search_tfidf
        #my_search.__init__(my_search)
        self.clear_dict()
        #my_search.search_tfidf_cosine(query,final_dict)
        indexing.Indexing.search_solr(query,final_dict)
        print("query time")
        t = time.time() - t1
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        print(final_dict)
        return {'journals': final_dict}, 201

api.add_resource(Main, '/')

if __name__ == '__main__':
    data_preparation = indexing.Indexing
    data_preparation.__init__(data_preparation)
    app.run(port=5000)
    #app.run(port=5000, debug=True)
