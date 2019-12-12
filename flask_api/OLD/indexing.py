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

stopWords = set(stopwords.words('english'))
stopWords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '/', '%'])
filename = 'articles.json'
searchItems = []

# ordered dictionary to save title and abstracts

global articles, documents
articles={}
documents = []
all_Of = []
articles=OrderedDict()
articles.setdefault("title", [])
articles.setdefault("abstract", [])
articles.setdefault("journal", [])
articles.setdefault("neighbors", [])
articles.setdefault("neighborsSimilarity", [])
# mean reciprocal rank
articles.setdefault("mrr", [])

# tfidf and cosine similarity matrix
global tfidf_matrix
global ddsim_matrix
# journals

# article limit
limit = 10
specialLimit = limit+1
similart = 0

# list to store titles and abstracts of every article in order to use it in tf-idf
documents = []
l_documents = []
documents1 = []
global invertedIndex
invertedIndex = {}
inverted_index = defaultdict(lambda: dict())
global query
global queryTerms, myquery
queryTerms = []
getDoc = []

newDocuments = []
final_score = {}

limit =10000
# top 100 neighbors
top100 = 100
# k nearest neighbors
k = 10
# now_articles= 100000
# all_articles=12.834.585
global t11,t


class Indexing(object):

    def __init__(self):
        self.myread(Indexing)
        #self.index(Indexing)
        #print("start")
        #self.solr_testing_repeat(Indexing)
        '''
        t11 = time.time()
        self.paok(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        '''
        #self.glove(Indexing)
        #self.word2vec(Indexing)
        #t11 = time.time()
        #self.myread(Indexing)
        # self.readfile(Indexing)
        # self.prepare(Indexing)
        # self.invert(Indexing)
        # self.documents_length(Indexing)
        #print('Indexing for limit ')
        #print(limit)
        '''
        t11 = time.time()
        #self.solr_indexing(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        t11 = time.time()
        print('query 1')
        #self.query_test(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        '''
        '''
        t11 = time.time()
        print('query 2')
        #self.query_test2(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        t11 = time.time()
        print('query 3')
        #self.query_test3(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        t11 = time.time()
        print('query 4')
        #self.query_test4(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        print('query 5')
        #self.query_test5(Indexing)
        #self.glove(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        t11 = time.time()
        print('query 6')
        #self.query_test6(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        t11 = time.time()
        print('query 7')
        #self.query_test7(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        t11 = time.time()
        print('query 8')
        #self.query_test8(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        print('query 9')
        #self.query_test9(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        t11 = time.time()
        print('query 10')
        #self.query_test10(Indexing)
        t = time.time() - t11
        print(time.strftime("%H:%M:%S", time.gmtime(t)))
        '''

    def index(self):

        ''' Indexing '''
        t11 = time.time()
        print('--- Indexing ---')
        counter = 0
        second_counter = 0
        solr = pysolr.Solr('http://localhost:8983/solr/articlescollection')
        solr.delete(q='*:*')
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter == limit:
                    break
                counter = counter + 1
                second_counter = second_counter + 1
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
                field = 'text_custom'
                temp_dict[field] = temp_word
                all_Of.append(temp_dict)
                if second_counter == 1000:
                    solr.add(all_Of)
                    all_Of.clear()
                    second_counter = 0
            solr.add(all_Of)
        data_file.close()
        t = time.time() - t11
        index_time = (time.strftime("%H:%M:%S", time.gmtime(t)))


    
    def search_solr(query,final_dict):
        print('paok')
        mystring= query
        mynewstring=tokenizer.create_new_string(mystring)
        url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom&fl=*,score&q='
        url+= mynewstring
        url+= '&rows=101&wt=json'
        data = urlopen(url)
        connection = data
        response = simplejson.load(connection)
        top_k = {}
        step = 1
        j_names = []
        final_rank = {}
        final_rank = OrderedDict()
        all_results = []
        for document in response['response']['docs']:
            temp_dict = {}
            temp_dict["id"] = document['id']
            temp_dict["title"] = str(document['title'])[2:-2]
            temp_dict["journal"] = str(document['journal'])[2:-2]
            temp_dict["abstract"] = str(document['abstract'])[2:-2]
            temp_dict["score"] = document['score']
            all_results.append(temp_dict)
            top_k[temp_dict["id"]] = temp_dict["score"]
            j_names.append(temp_dict["journal"])
            final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
        for f_r in final_rank:
            temp_dict = {}
            temp_dict["journal"] = f_r
            final_dict.append(temp_dict)

        


        












    def paok(self):
        counter = 0
        limit = 100000
        all_words = []
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
                        #if w not in all_words:
                        all_words.append(w)

        #print(len(all_words))
        all_words=list(set(all_words))
        wordnet_lemmatizer = WordNetLemmatizer()    
        #print(len(all_words))
        file_out = open('100000LemmSynList.txt', "w")
        for word in all_words:
            l_word = wordnet_lemmatizer.lemmatize(word,pos='v')
            if l_word != word:
                line = word + ' => ' + l_word + '\n'
                file_out.write(line)
        file_out.close()

    def glove(self):
        word_embeddings.glove()

    def word2vec(self):
        word_embeddings.my_word2vec()

    def myread(self):
        counter = 0
        limit = 10000
        all_journals = []
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter == limit:
                    break 
                counter = counter + 1
                data = line[:-2]
                data = json.loads(data)
                if(data["year"] == None):
                    all_journals.append('0000')
                else:
                    all_journals.append(data["year"])
        c = Counter(all_journals)        
        print(c)

        articles_counter = {}
        for journal in  all_journals:
            if journal in articles_counter:
                articles_counter[journal] += 1
            else:
                articles_counter[journal] = 1
        #articles_counter = reversed(sorted(articles_counter.items(), key=operator.itemgetter(1)))
        #articles_counter = sorted(articles_counter.items(), key=operator.itemgetter(0))
        file_out = open(str(limit) + 'ArticlesYearByYear.txt', "w")
        for key, value in articles_counter.items():
            line = key + ' ' + str(value) + ' times \n'
            file_out.write(line)
        file_out.close()
        



    def solr_testing_repeat(self):
        field_list = []

        
        '''
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
        '''
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
        print(limit)
        for field in field_list:
            print(field_list.index(field)+1," of ",len(field_list))
            print("Method ",field)
            t11 = time.time()
            Indexing.my_ultimate_test(field)




    def my_ultimate_test(field):
        ''' Indexing '''
        t11 = time.time()
        print('--- Indexing ---')
        counter = 0
        second_counter = 0
        solr = pysolr.Solr('http://localhost:8983/solr/articlescollection')
        solr.delete(q='*:*')
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter == limit:
                    break
                counter = counter + 1
                second_counter = second_counter + 1
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
                if second_counter == 1000:
                    solr.add(all_Of)
                    all_Of.clear()
                    second_counter = 0
            solr.add(all_Of)
        data_file.close()
        t = time.time() - t11
        index_time = (time.strftime("%H:%M:%S", time.gmtime(t)))
        

        ''' Querying '''
        t1 = time.time()
        print('--- Querying ---')
        method=field
        #clusters = ''
        #myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        myfilename = method+'_KStem_output_' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
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
                query = Q(t=mystring)
                query = str(query)[2:]
                #print(query)
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
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t1
        query_time = (time.strftime("%H:%M:%S", time.gmtime(t)))
        myprint.print_time_MRR(mrr,myfilename,index_time,query_time)
        myprint.print_MRR_Time_comparison(mrr,method,limit,index_time,query_time)


    def solr_indexing(self):
        counter = 0
        second_counter = 0
        solr = pysolr.Solr('http://localhost:8983/solr/articlescollection')
        solr.delete(q='*:*')
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter == limit:
                    break
                counter = counter + 1
                second_counter = second_counter + 1
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
                '''
                #temp_dict["text_en_splitting"] = tokenizer.create_new_string(temp_dict["title"] + ' ' + temp_dict["abstract"])
                temp_dict["text_en_splitting"] = temp_dict["title"] + ' ' + temp_dict["abstract"]
                temp_dict["text_hunspell_stem"] = temp_dict["title"] + ' ' + temp_dict["abstract"]
                temp_dict["text_porter_stem"] = temp_dict["title"] + ' ' + temp_dict["abstract"]
                temp_dict["text_kstem_stem"] = temp_dict["title"] + ' ' + temp_dict["abstract"]
                temp_dict["text_snowball_stem"] = temp_dict["title"] + ' ' + temp_dict["abstract"]
                '''
                
                #temp_word =tokenizer.create_new_string2(temp_dict["title"] + ' ' + temp_dict["abstract"])
                temp_word =temp_dict["title"] + ' ' + temp_dict["abstract"]
                '''
                temp_word = temp_dict["abstract"]
                weight_title = tokenizer.create_new_string_title(temp_dict["title"])
                if weight_title == 0:
                    temp_word1 = temp_word
                else:
                    temp_word1 = weight_title + ' ' + temp_word
                '''
                #temp_dict["text_custom"] = temp_word
                #temp_dict["text_general"] = temp_word1
                #temp_dict["text_custom_word2vec"] = temp_word
                #temp_dict["text_custom_word2vec2"] = temp_word
                #temp_dict["text_custom_word2vec3"] = temp_word
                #temp_dict["text_custom_word2vec4"] = temp_word
                #temp_dict["text_custom_word2vec5"] = temp_word
                #temp_dict["text_custom_glove"] = temp_word
                #temp_dict["text_custom_glove2"] = temp_word
                #temp_dict["text_custom_glove3"] = temp_word
                #temp_dict["text_custom_glove4"] = temp_word
                #temp_dict["text_custom_glove5"] = temp_word
                #temp_dict["text_custom_kstem"] = temp_word1
                #temp_dict["text_custom_hunspellstem2"] = temp_word
                #temp_dict["text_custom_porterstem2"] = temp_word
                temp_dict["text_custom_kstem2"] = temp_word
                #temp_dict["text_custom_snowballporterstem2"] = temp_word
                #temp_dict["text_custom"] = tokenizer.create_lemmatization(temp_word)

                all_Of.append(temp_dict)
                if second_counter == 1000:
                    solr.add(all_Of)
                    all_Of.clear()
                    print(counter)
                    second_counter = 0
            solr.add(all_Of)
            '''
            strint = tokenizer.create_new_string2("Efficacy of the Herpes Zoster Subunit Vaccine in Adults 70 Years of Age or Older. BACKGROUND: A trial involving adults 50 years of age or older (ZOE-50) showed that the herpes zoster subunit vaccine (HZ/su) containing recombinant varicella-zoster virus glycoprotein E and the AS01B adjuvant system was associated with a risk of herpes zoster that was 97.2% lower than that associated with placebo. A second trial was performed concurrently at the same sites and examined the safety and efficacy of HZ/su in adults 70 years of age or older (ZOE-70).METHODS: This randomized, placebo-controlled, phase 3 trial was conducted in 18 countries and involved adults 70 years of age or older. Participants received two doses of HZ/su or placebo (assigned in a 1:1 ratio) administered intramuscularly 2 months apart. Vaccine efficacy against herpes zoster and postherpetic neuralgia was assessed in participants from ZOE-70 and in participants pooled from ZOE-70 and ZOE-50.RESULTS: In ZOE-70, 13,900 participants who could be evaluated (mean age, 75.6 years) received either HZ/su (6950 participants) or placebo (6950 participants). During a mean follow-up period of 3.7 years, herpes zoster occurred in 23 HZ/surecipients and in 223 placebo recipients (0.9 vs. 9.2 per 1000 person-years). Vaccine efficacy against herpes zoster was 89.8% (95% confidence interval [CI], 84.2 to 93.7;P<0.001) and was similar in participants 70 to 79 years of age (90.0%) and participants 80 years of age or older (89.1%). In pooled analyses of data from participants 70 years of age or older in ZOE-50 and ZOE-70 (16,596 participants), vaccine efficacy against herpes zoster was 91.3% (95% CI, 86.8 to 94.5; P<0.001), and vaccine efficacy against postherpetic neuralgia was 88.8% (95% CI, 68.7 to 97.1; P<0.001). Solicited reports of injection-site and systemic reactions within 7 days after injection were more frequent among HZ/su recipients than among placebo recipients (79.0% vs. 29.5%). Serious adverse events, potential immune-mediated diseases, and deaths occurred with similar frequencies in the two study groups.CONCLUSIONS: In our trial, HZ/su was found to reduce the risks of herpes zoster and postherpetic neuralgia among adults 70 years of age or older. (Funded by GlaxoSmithKline Biologicals; ZOE-50 and ZOE-70 ClinicalTrials.gov numbers, NCT01165177 and NCT01165229 .).")
            print(strint)
            results = solr.search(strint, **{
                'df': 'text_custom',
                'fl': '*,score',
                'rows': '101'
                })
            #results = solr.search('*:*')
            print(results.raw_response['response']['numFound'])
            for result in results:
                print("The journal is '{0}'.".format(result['journal']))
                print("The score is '{0}'.".format(result['score']))
            '''
        
    
    def query_test(self):

        solr = pysolr.Solr('http://localhost:8983/solr/articlescollection')
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='text_custom_kstem2'
        clusters = ''
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                '''
                
                temp1 = tokenizer.create_new_string_title2(data["title"])
                temp_word = data["title"] + ' ' + data["abstractText"]
                #temp_word = data["abstractText"]
                temp2 = tokenizer.create_new_string(temp_word)
                if temp1 == 0:
                    mystring2 = temp2
                else:
                    mystring2 = temp1+temp2
                #print(mystring2)
                '''
                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string22(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_kstem2&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #mynewstring=urllib.parse.quote(mynewstring)
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                #print(url)
                '''
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                print(response['response']['numFound'], "documents found.")
                '''
                #print(mynewstring)
                '''
                solr2 = SolrClient('http://localhost:8983/solr')
                res = solr2.query('articlescollection',{
                    'q': mynewstring,
                    'df': 'text_custom',
                    'fl': '*,score',
                    'rows': '101'
                    })
                print(res.get_results_count())
                #print(res.docs)
                '''
                query = Q(t=mystring)
                query = str(query)[2:]
                #print(query)
                documents = solr.search(query, **{
                    'df': 'text_custom_kstem2',
                    'fl': '*,score',
                    'rows': '101'
                    })
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
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
                '''
                #connection = data
                #response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                '''
                step = 1
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)

    def query_test2(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='W2V'
        clusters = 2500
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                temp1 =tokenizer.create_new_string_title(data["title"])
                temp2 = tokenizer.create_new_string(data["abstractText"])
                mystring2 = temp1+temp2

                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_word2vec2&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)

    def query_test3(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='W2V'
        clusters = 3000
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                temp1 =tokenizer.create_new_string_title(data["title"])
                temp2 = tokenizer.create_new_string(data["abstractText"])
                mystring2 = temp1+temp2

                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_word2vec3&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)

    def query_test4(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='W2V'
        clusters = 3500
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                temp1 =tokenizer.create_new_string_title(data["title"])
                temp2 = tokenizer.create_new_string(data["abstractText"])
                mystring2 = temp1+temp2

                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_word2vec4&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)

    def query_test5(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='W2V'
        clusters = 5000
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                temp1 =tokenizer.create_new_string_title(data["title"])
                temp2 = tokenizer.create_new_string(data["abstractText"])
                mystring2 = temp1+temp2

                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_word2vec5&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)

    def query_test6(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='glove'
        clusters = 6000
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                temp1 =tokenizer.create_new_string_title(data["title"])
                temp2 = tokenizer.create_new_string(data["abstractText"])
                mystring2 = temp1+temp2

                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_glove&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)

    def query_test7(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='glove'
        clusters = 2500
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                temp1 =tokenizer.create_new_string_title(data["title"])
                temp2 = tokenizer.create_new_string(data["abstractText"])
                mystring2 = temp1+temp2

                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_glove2&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)

    def query_test8(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='glove'
        clusters = 3000
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                temp1 =tokenizer.create_new_string_title(data["title"])
                temp2 = tokenizer.create_new_string(data["abstractText"])
                mystring2 = temp1+temp2

                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_glove3&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)

    def query_test9(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='glove'
        clusters = 3500
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                #temp1 =tokenizer.create_new_string_title(data["title"])
                #temp2 = tokenizer.create_new_string(data["abstractText"])
                #mystring2 = temp1+temp2

                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_glove4&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)

    def query_test10(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        #myfilename = 'Hunspell2_output' + str(limit)
        method='glove'
        clusters = 5000
        myfilename = method+'_'+str(clusters)+'_output' + str(limit)
        #query = 'Long-term Recurrence and Complications Associated With Elective Incisional Hernia Repair. Importance: Prosthetic mesh is frequently used to reinforce the repair of abdominal wall incisional hernias. The benefits of mesh for reducing the risk of hernia recurrence or the long-term risks of mesh-related complications are not known.Objective: To investigate the risks of long-term recurrence and mesh-related complications following elective abdominal wall hernia repair in a population with complete follow-up.Design, Setting, and Participants: Registry-based nationwide cohort study including all elective incisional hernia repairs in Denmark from January 1, 2007, to December 31, 2010. A total of 3242 patients with incisional repair were included. Follow-up until November 1, 2014, was obtained by merging data with prospective registrations from the Danish National Patient Registry supplemented with a retrospective manual review of patient records. A 100% follow-up rate was obtained.Exposures: Hernia repair using mesh performed by either open or laparoscopic techniques vs open repair without use of mesh.Main Outcomes and Measures: Five-year risk of reoperation for recurrence and 5-year risk of all mesh-related complications requiring subsequent surgery.Results: Among the 3242 patients (mean age, 58.5 [SD, 13.5] years; 1720 women [53.1%]), 1119 underwent open mesh repair (34.5%), 366 had open nonmesh repair (11.3%), and 1757 had laparoscopic mesh repair (54.2%). The median follow-up after open mesh repair was 59 (interquartile range [IQR], 44-80) months, after nonmesh open repair was 62 (IQR, 44-79) months, and after laparoscopic mesh repair was 61 (IQR, 48-78) months. The risk of the need for repair for recurrent hernia following these initial hernia operations was lower for patients with open mesh repair (12.3% [95% CI, 10.4%-14.3%]; risk difference, -4.8% [95% CI, -9.1% to -0.5%]) and for patients with laparoscopic mesh repair (10.6% [95% CI, 9.2%-12.1%]; risk difference, -6.5% [95% CI, -10.6% to -2.4%]) compared with nonmesh repair (17.1% [95% CI, 13.2%-20.9%]). For the entirety of the follow-up duration, there was a progressively increasing number of mesh-related complications for both open and laparoscopic procedures. At 5 years of follow-up, the cumulative incidence of mesh-related complications was 5.6% (95% CI, 4.2%-6.9%) for patients who underwent open mesh hernia repair and 3.7% (95% CI, 2.8%-4.6%) for patients who underwent laparoscopic mesh repair. The long-term repair-related complication rate for patients with an initial nonmesh repair was 0.8% (open nonmesh repair vs open mesh repair: risk difference, 5.3% [95% CI, 4.4%-6.2%]; open nonmesh repair vs laparoscopic mesh repair: risk difference, 3.4% [95% CI, 2.7%-4.1%]).Conclusions and Relevance: Among patients undergoing incisional repair, sutured repair was associated with a higher risk of reoperation for recurrence over 5 years compared with open mesh and laparoscopic mesh repair. With long-term follow-up, the benefits attributable to mesh are offset in part by mesh-related complications.'
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                search_query = data["title"] + ' ' + data["abstractText"]
                #print(data["title"] + ' ' + data["abstractText"])
                #search_query = query
                search_journal = data["journal"]
                search_id = data["pmid"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                temp1 =tokenizer.create_new_string_title(data["title"])
                temp2 = tokenizer.create_new_string(data["abstractText"])
                mystring2 = temp1+temp2

                #mystring2 = aces + temp2
                #mynewstring=tokenizer.create_lemmatization(mystring)
                mynewstring=tokenizer.create_new_string(mystring)
                #mynewstring = mystring2
                #print(mynewstring)
                #'http://localhost:8983/solr/articlescollection/select?defType=dismax&df=title,abstract&fl=journal,score&q=long&qf=title^2&rows=101&wt=json'
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_hunspellstem2&fl=*,score&q='
                url = 'http://localhost:8983/solr/articlescollection/select?'
                #url+= 'defType=edismax&'
                #url+= 'df=title,text_general&fl=*,score&q='
                url+= 'df=text_custom_glove5&fl=*,score&q='
                #url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_word2vec&fl=*,score&q='
                url+= mynewstring
                #url+='&qf=title^2%20text_general'
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
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
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,method+'_'+str(clusters),limit)


'''
    def query_test2(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        myfilename = 'Porter2_output' + str(limit)
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                #search_query = tokenizer.create_new_string(data["title"] + ' ' + data["abstractText"])
                search_query = data["title"] + ' ' + data["abstractText"]
                temps = data["journal"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                mynewstring=tokenizer.create_new_string(mystring)
                url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_porterstem2&fl=*,score&q='
                url+= mynewstring
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
                    temp_dict = {}
                    temp_dict["id"] = document['id']
                    temp_dict["title"] = str(document['title'])[2:-2]
                    temp_dict["journal"] = str(document['journal'])[2:-2]
                    temp_dict["abstract"] = str(document['abstract'])[2:-2]
                    temp_dict["score"] = document['score']
                    if step != 1: 
                        all_results.append(temp_dict)
                        top_k[temp_dict["id"]] = temp_dict["score"]
                        j_names.append(temp_dict["journal"])
                    else:
                        old_dic.clear()
                        old_dic = temp_dict
                        step+=1
                step = 1
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,'Porter2',limit)
    
    def query_test3(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        myfilename = 'KStem2_output' + str(limit)
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                #search_query = tokenizer.create_new_string(data["title"] + ' ' + data["abstractText"])
                search_query = data["title"] + ' ' + data["abstractText"]
                temps = data["journal"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                mynewstring=tokenizer.create_new_string(mystring)
                url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_kstem2&fl=*,score&q='
                url+= mynewstring
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
                    temp_dict = {}
                    temp_dict["id"] = document['id']
                    temp_dict["title"] = str(document['title'])[2:-2]
                    temp_dict["journal"] = str(document['journal'])[2:-2]
                    temp_dict["abstract"] = str(document['abstract'])[2:-2]
                    temp_dict["score"] = document['score']
                    if step != 1: 
                        all_results.append(temp_dict)
                        top_k[temp_dict["id"]] = temp_dict["score"]
                        j_names.append(temp_dict["journal"])
                    else:
                        old_dic.clear()
                        old_dic = temp_dict
                        step+=1
                step = 1
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,'Kstem2',limit)
    
    def query_test4(self):
        #myfilename = "output1000pt"
        #myfilename = "output10000s"
        myfilename = 'Snowball2_output' + str(limit)
        try:
            os.remove(myfilename)
        except OSError:
            pass
        score_dict = {}
        counter1 = 0;
        mrr = 0;
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            for line in data_file:
                if counter1 == limit:
                    break
                counter1 = counter1 + 1
                data = line[:-2]
                data = json.loads(data)
                #search_query = tokenizer.create_new_string(data["title"] + ' ' + data["abstractText"])
                search_query = data["title"] + ' ' + data["abstractText"]
                temps = data["journal"]
                old_dic = {}
                old_dic["id"] = data["pmid"]
                old_dic["title"] = data["title"]
                old_dic["journal"] = data["journal"]
                mystring= search_query
                mynewstring=tokenizer.create_new_string(mystring)
                url = 'http://localhost:8983/solr/articlescollection/select?df=text_custom_snowballporterstem2&fl=*,score&q='
                url+= mynewstring
                #print(data["pmid"])
                #print(data["journal"])
                url+= '&rows=101&wt=json'
                data = urlopen(url)
                connection = data
                response = simplejson.load(connection)
                #print(response['response']['numFound'], "documents found.")
                # Print the name of each document.
                top_k = {}
                step = 1
                j_names = []
                final_rank = {}
                final_rank = OrderedDict()
                all_results = []
                for document in response['response']['docs']:
                    temp_dict = {}
                    temp_dict["id"] = document['id']
                    temp_dict["title"] = str(document['title'])[2:-2]
                    temp_dict["journal"] = str(document['journal'])[2:-2]
                    temp_dict["abstract"] = str(document['abstract'])[2:-2]
                    temp_dict["score"] = document['score']
                    if step != 1: 
                        all_results.append(temp_dict)
                        top_k[temp_dict["id"]] = temp_dict["score"]
                        j_names.append(temp_dict["journal"])
                    else:
                        old_dic.clear()
                        old_dic = temp_dict
                        step+=1
                step = 1
                #print(len(j_names))
                #print(j_names)
                final_rank = solr_sorting_mrr.get_final_rank(j_names,k)
                #print(final_rank)
                reciprocal_rank = solr_sorting_mrr.get_rr(final_rank,old_dic,k)
                mrr = mrr + reciprocal_rank
                #print(mean_reciprocal_rank)
                myprint.print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k)

            print(counter1)
        data_file.close()
        mrr = mrr / float(limit)
        t = time.time() - t11
        myprint.print_MRR(mrr,myfilename,(time.strftime("%H:%M:%S", time.gmtime(t))))
        myprint.print_MRR_comparison(mrr,'Snowball2',limit)
                

    def readfile(self):
        counter = 0
        with io.open('articles.json', 'r', encoding='utf-8', errors='ignore') as data_file:
            next(data_file)
            next(data_file)
            for line in data_file:
                if counter == limit:
                    break
                counter = counter + 1
                data = line[:-2]
                data = json.loads(data)
                articles["title"].append(data["title"])
                articles["abstract"].append(data["abstractText"])
                articles["journal"].append(data["journal"])
                # keep titles and abstracts without special characters

            

    def prepare(self):
        for x in range(limit):
            temp = articles["title"][x] + ' ' + articles["abstract"][x].lower()
            documents1.append(tokenizer.prepare(temp, documents))

    def invert(self):
        for numdoc in range(limit):
            termslist = documents[numdoc]
            for docterm in range(len(termslist)):
                word = documents[numdoc][docterm]
                global invertedIndex
                # print(word)
                if not invertedIndex.__contains__(word):
                    mypos = {}
                    mypos.__setitem__(numdoc, 1)
                    invertedIndex.__setitem__(word, mypos)
                else:
                    tempd = invertedIndex[word]
                    v = tempd.pop(numdoc, None)
                    if (v == None):
                        tempd[numdoc] = 1
                    else:
                        tempd[numdoc] = v + 1
                    invertedIndex.pop(word, None)
                    invertedIndex[word] = tempd

    def documents_length(self):
        for key in range(limit):
            unique_documents_terms = set(documents[key])
            unique_documents_terms = list(unique_documents_terms)
            Length = 0
            for pos in range(len(unique_documents_terms)):
                doc_term = invertedIndex[unique_documents_terms[pos]]
                tf = 1 + math.log(doc_term[key])
                Length += tf * tf
            length = math.sqrt(Length)
            l_documents.append(length)
'''
