# -*- coding: utf-8 -*-
import tokenizer
import word_embeddings
import codecs
import math
from operator import itemgetter
import heapq
import collections
from collections import OrderedDict, defaultdict
from nltk.corpus import stopwords
import re
import numpy as np
import time
import pysolr
from solrq import Q, Value
from SolrClient import SolrClient
from collections import OrderedDict
import json
import requests
import io,sys
from solrq import Q
from urllib.request import *
import urllib,urllib3
import simplejson
import numpy as np
import re
import time
import ast
import numpy as np
import collections
from collections import OrderedDict,Counter
import app
import sys
import solr_sorting_mrr
import myprint
import os
import gensim
import operator
from nltk.stem import WordNetLemmatizer
final_dict = app.final_dict

#  creation of stopwords list
stopWords = set(stopwords.words('english'))
stopWords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '/', '%'])

# basic variables initialization
global documents, filename, query, myquery, limit, t11, t
filename = 'articles.json'
documents = []
all_Of = []
# article limit
limit = 10
# top 100 neighbors
top100 = 100
# k nearest neighbors
k = 10
# now_articles= 100000
# all_articles=12.834.585


class Indexing(object):

    # init all the functions
    def __init__(self):
        self.index(Indexing)

        self.search_solr(Indexing,final_dict)
        '''
        self.create_lemm_file(Indexing)
        self.statistics(Indexing)
        self.word2vec(Indexing)
        self.glove(Indexing)
        self.solr_testing_repeat(Indexing)
        self.my_ultimate_test(Indexing)
        '''
        '''
        t11 = time.time()
        self.radndom(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        '''
    
    # this function helps us when we just want to index some documents in our solr installation
    def index(self):
        ''' Indexing '''

        # time, variables, solr initialization and delete of tehe previous records.
        t11 = time.time()
        print('--- Indexing ---')
        counter = 0
        second_counter = 0
        solr = pysolr.Solr('http://localhost:8983/solr/articlescollection')
        solr.delete(q='*:*')

        # reading from the articles file
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                
                # break for articles limit
                if counter == limit:
                    break
                counter = counter + 1
                second_counter = second_counter + 1

                # add all the fields of an article to a dictionary and import it to the list of the records
                data = line[:-2]
                data = json.loads(data)
                temp_dict = {}
                temp_dict["id"] = data["pmid"]
                #temp_dict["title"] = data["title"]
                #temp_dict["title_st"] = data["title"]
                temp_dict["journal"] = data["journal"]
                #temp_dict["abstract"] = data["abstractText"]
                #temp_dict["meshMajor"] = data["meshMajor"]
                #temp_dict["year"] = data["year"]
                temp_word = data["title"] + ' ' + data["abstractText"]

                # declaration of the field where we will index the records
                field = 'text_custom'
                temp_dict[field] = temp_word
                all_Of.append(temp_dict)

                # just a control of ram usage when we import the records to the list
                if second_counter == 10000:
                    print(counter)
                    solr.add(all_Of)
                    all_Of.clear()
                    second_counter = 0
            
            # add the records list to solr
            solr.add(all_Of)
        
        # close file and terminate the work
        data_file.close()
        t = time.time() - t11
        index_time = (time.strftime("%H:%M:%S", time.gmtime(t)))

    
    # this function is used when we want to find the suggested journals in an already created collection in solr, 
    # by using the frontend part of our application
    def search_solr(query,final_dict):
        solr = pysolr.Solr('http://localhost:8983/solr/articlescollection')

        # here the query string will be escaped to ensure that final query string will not be broken by some rougue search value.
        # also we make the search in solr collection and store results to documents list
        mystring = query
        myquery = Q(t=mystring)
        myquery = str(myquery)[2:]
        documents = solr.search(myquery, **{
            'df': 'text_custom',
            'fl': '*,score',
            'rows': '101',
            'wt': 'json'
            })
        
        # After the declarations, we save alla the journals name in j_names list and we keep the top k (10 in our case)
        # in final_rank. Finaly we store every journal of the results as a dictionary (so as to keep more fields than 
        # the journal name from the response if we want), in a list, which we return.
        j_names = []
        final_rank = {}
        final_rank = OrderedDict()
        for document in documents:
            j_names.append(str(document['journal'])[2:-2])
        final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
        for f_r in final_rank:
            temp_dict = {}
            temp_dict["journal"] = f_r
            final_dict.append(temp_dict) 
    
    # this function is about the creation of a lemmatization file from the vocabular of the articles 
    def create_lemm_file(self):
        counter = 0
        limit = 100000
        all_words = []

        # just file reading and save every word that is not a stopword in all_words list
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter == limit:
                    break 
                counter = counter + 1
                data = line[:-2]
                data = json.loads(data)
                myline = data["title"] + ' ' + data["abstractText"]
                words = gensim.utils.simple_preprocess(myline)
                wordsFiltered = []
                for w in words:
                    if w not in stopWords:
                        all_words.append(w)
                    
        # keep unique entries, lemmatize and save with specific format in file
        all_words=list(set(all_words))
        wordnet_lemmatizer = WordNetLemmatizer() 
        file_out = open(str(limit)+'LemmSynList.txt', "w")
        for word in all_words:
            l_word = wordnet_lemmatizer.lemmatize(word,pos='v')
            if l_word != word:
                line = word + ' => ' + l_word + '\n'
                file_out.write(line)
        file_out.close()

    # call glove operation
    def glove(self):
        word_embeddings.my_glove()

    # call word2vec operation
    def word2vec(self):
        word_embeddings.my_word2vec()

    # a function to write down some statistics about the number of appearances of every journal, the number of journals
    # and the chronological distribution, for specific number of articles.
    def statistics(self):
        counter = 0
        limit = 10000
        all_journals = []
        # file reading and storing of years or journals
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter == limit:
                    break 
                counter = counter + 1
                data = line[:-2]
                data = json.loads(data)
                # all_journals.append(data["journal"])
                if(data["year"] == None):
                    all_journals.append('0000')
                else:
                    all_journals.append(data["year"])
        
        # counting the appearances and recording in a file
        articles_counter = {}
        for journal in  all_journals:
            if journal in articles_counter:
                articles_counter[journal] += 1
            else:
                articles_counter[journal] = 1
        # articles_counter = reversed(sorted(articles_counter.items(), key=operator.itemgetter(1)))
        # articles_counter = sorted(articles_counter.items(), key=operator.itemgetter(0))
        file_out = open(str(limit) + 'ArticlesYearByYear.txt', "w")
        # file_out = open(str(limit) + 'ArticlesYearByJournal.txt', "w")
        for key, value in articles_counter.items():
            line = key + ' ' + str(value) + ' times \n'
            file_out.write(line)
        file_out.close()

    # main function for test purposes.
    def solr_testing_repeat(self):

        # saving in a list alla the fields in which, we will make the tests
        field_list = []
        field_list.append('text_custom')
        field_list.append('text_custom_kstem')
        field_list.append('text_custom_porterstem')
        field_list.append('text_custom_snowballporterstem')
        field_list.append('text_custom_hunspellstem')
        field_list.append('text_custom_kstem2')
        field_list.append('text_custom_porterstem2')
        field_list.append('text_custom_snowballporterstem2')
        field_list.append('text_custom_hunspellstem2')
        field_list.append('text_custom_lemm_stock')
        field_list.append('text_custom_lemm_list')
        field_list.append('text_custom_lemm_w2v')
        field_list.append('text_custom_lemm_glove')
        field_list.append('text_custom_word2vec')
        field_list.append('text_custom_word2vec2')
        field_list.append('text_custom_word2vec3')
        field_list.append('text_custom_word2vec4')
        field_list.append('text_custom_word2vec5')
        field_list.append('text_custom_glove')
        field_list.append('text_custom_glove2')
        field_list.append('text_custom_glove3')
        field_list.append('text_custom_glove4')
        field_list.append('text_custom_glove5')
        #print(len(field_list))
        global limit
        limit = 10000

        # in iteration we call the my_ultimate_test function to make indexing and querying for all the cases
        for field in field_list:
            print(field_list.index(field)+1," of ",len(field_list))
            print("Method ",field)
            t11 = time.time()
            Indexing.my_ultimate_test(field)

    # it's the function in which we do the indexing and querying for every case of the testing loop
    def my_ultimate_test(field):
        ''' Indexing '''
        t11 = time.time()
        print('--- Indexing ---')
        counter = 0
        second_counter = 0
        solr = pysolr.Solr('http://localhost:8983/solr/articlescollection')
        solr.delete(q='*:*')

        # after some declarations, we read the file of the articles create a dictionary to keep all the data,
        # store them temporarly in the list all_of, and with 
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:

                # we check with this counter if we are under the limit of articles
                if counter == limit:
                    break
                counter = counter + 1
                second_counter = second_counter + 1

                # we create a temp dictionary to keep all the data and store them temporarly in the list all_of
                data = line[:-2]
                data = json.loads(data)
                temp_dict = {}
                temp_dict["id"] = data["pmid"]
                temp_dict["title"] = data["title"]
                temp_dict["title_st"] = data["title"]
                temp_dict["journal"] = data["journal"]
                temp_dict["abstract"] = data["abstractText"]
                temp_dict["meshMajor"] = data["meshMajor"]
                temp_dict["year"] = data["year"]
                temp_word = temp_dict["title"] + ' ' + temp_dict["abstract"]
                if field == 'text_custom_lemm_stock':
                    temp_dict[field] = tokenizer.create_lemmatization(temp_word)
                else:
                    temp_dict[field] = temp_word
                all_Of.append(temp_dict)

                # we use second_counter to control ram usage and add data for every 1000 articles to solr
                if second_counter == 1000:
                    solr.add(all_Of)
                    all_Of.clear()
                    second_counter = 0

        # just add remaining articles to solr and do the last work for indexing
            solr.add(all_Of)
        data_file.close()
        t = time.time() - t11
        index_time = (time.strftime("%H:%M:%S", time.gmtime(t)))
        
        ''' Querying '''
        t1 = time.time()
        print('--- Querying ---')

        # declaration of methon which is depended on field name that we have as import
        method=field

        # any clusters declarations and creation of filename for results, with remiving of the any previours record.
        #clusters = ''
        #myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        myfilename = method+'_KStem_output_' + str(limit)
        try:
            os.remove(myfilename)
        except OSError:
            pass
        counter1 = 0;
        mrr = 0;

        # read entries of the articles file to use them a search examples
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                if field == 'text_custom_lemm_stock':
                    mystring = tokenizer.create_lemmatization(search_query)
                else:
                    mystring = search_query
                
                # proper preparation of query parsing
                query = Q(t=mystring)
                query = str(query)[2:]

                # storing results in dicuments list 
                documents = solr.search(query, **{
                    'df': method,
                    'fl': '*,score',
                    'rows': '101',
                    'wt': 'json'
                    })
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []

                # for every returned result, we ignore the same article as that in index by checking the id
                for document in documents:
                    temp_dict = {}
                    temp_dict["id"] = document['id']
                    temp_dict["title"] = str(document['title'])[2:-2]
                    temp_dict["journal"] = str(document['journal'])[2:-2]
                    temp_dict["abstract"] = str(document['abstract'])[2:-2]
                    temp_dict["score"] = document['score']
                    if search_id != temp_dict["id"]: 
                        all_results.append(temp_dict)
                        top_k[temp_dict["id"]] = temp_dict["score"]
                        j_names.append(temp_dict["journal"])
                    else:
                        old_dic.clear()
                        old_dic = temp_dict
                        step+=1
                step = 1

                # final rank preparation and computation of reciprocal rank and the mean reciprocal rank
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)
        
        # final work and recording of mmr in specific result file and in general comparison file
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t1
        query_time = (time.strftime("%H:%M:%S", time.gmtime(t)))
        myprint.print_time_MRR(mrr,myfilename,index_time,query_time)
        myprint.print_MRR_Time_comparison(mrr,method,limit,index_time,query_time)
