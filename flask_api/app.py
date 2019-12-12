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
global data_preparation,my_search


# init flask
app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# list to sabe resutls
final_dict=[]

class Main(Resource):

    # function to clear the list of journals between the searches
    def clear_dict(self):
        final_dict.clear()   
     
    # The function where i get and prepare the fields of title and abstract
    # from the frontend part and prepare them so as to use them in backend.
    # In the end, i return the suggested journals.
    def post(self):
        t1 = time.time()

        # getting the needed (title and abstract fields)
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

        # prepare them so as to use them in backend
        data = parser.parse_args()
        searchData = {'title': data['title'], 'abstract': data['abstract']}
        query = searchData['title'] + ' ' + searchData['abstract']
        self.clear_dict()

        # calling the search function and return the results.
        indexing.Indexing.search_solr(query,final_dict)
        return {'journals': final_dict}, 201


# Here i do the initialization of the app
api.add_resource(Main, '/')
if __name__ == '__main__':
    data_preparation = indexing.Indexing
    data_preparation.__init__(data_preparation)
    app.run(port=5000)
    #app.run(port=5000, debug=True)
