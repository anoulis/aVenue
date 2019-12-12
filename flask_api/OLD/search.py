# -*- coding: utf-8 -*-
import tokenizer
import codecs
import math
import collections
from collections import OrderedDict, defaultdict
from nltk.corpus import stopwords
import re
import numpy as np
import time
from collections import OrderedDict,Counter
import indexing
import meanReciprocalRank
import myprint
import app

invertedIndex = indexing.invertedIndex
limit = indexing.limit
documents = indexing.documents
k = indexing.k
articles = indexing.articles
documents_length = indexing.l_documents
final_dict = app.final_dict

class Search_tfidf():

    def __init__(self):
        global queryTerms, myquery, top_k, final_rank, mean_reciprocal_rank, final_score
        final_rank = {}
        final_rank = OrderedDict()
        top_k = {}
        final_score = {}
        queryTerms = []

    def search_tfidf(query):
        myquery = tokenizer.prepare(query, queryTerms)
        terms = queryTerms[0]
        terms = list(set(terms))
        for term in terms:
            if invertedIndex.__contains__(term):
                mykeys = list(invertedIndex[term].keys())
                myvalues = list(invertedIndex[term].values())
                idf = math.log((float(limit) / float(len(mykeys))) + 1)
                for doc in range(len(mykeys)):
                    tf = float(myvalues[doc]) / float(len(documents[mykeys[doc]]))
                    if mykeys[doc] not in final_score.keys():
                        final_score[mykeys[doc]] = float(tf * idf)
                    final_score[mykeys[doc]] += float((tf * idf))
        top_k = Counter(final_score)
        top_k = top_k.most_common(k)
        final_rank = meanReciprocalRank.get_final_rank(top_k,k)
        mean_reciprocal_rank = meanReciprocalRank.get_mrr()
        myprint.print_results(final_rank, mean_reciprocal_rank)

    def search_tfidf_cosine(query,final_dict):
        myquery = tokenizer.prepare(query, queryTerms)
        terms = queryTerms[0]
        terms = list(set(terms))
        for term in terms:
            if invertedIndex.__contains__(term):
                mykeys = list(invertedIndex[term].keys())
                myvalues = list(invertedIndex[term].values())
                idf = math.log(float(1 + limit) / float(len(mykeys)))
                for doc in range(len(mykeys)):
                    tf = 1 + math.log(myvalues[doc])
                    if mykeys[doc] not in final_score.keys():
                        final_score[mykeys[doc]] = float(tf * idf)
                    final_score[mykeys[doc]] += float((tf * idf))

        for key in final_score:
            final_score[key] = float(final_score[key]) / float(documents_length[key])
        top_k = Counter(final_score)
        top_k = top_k.most_common(k)
        #myprint.print_k_top(top_k)
        final_rank = meanReciprocalRank.get_final_rank(top_k,k)
        mean_reciprocal_rank = meanReciprocalRank.get_mrr()
       # myprint.print_results(final_rank, mean_reciprocal_rank)

        for key,value in final_rank:
            temp_dict = {}
            temp_dict["journal"] = key
            temp_dict["score"] = value
            final_dict.append(temp_dict)
