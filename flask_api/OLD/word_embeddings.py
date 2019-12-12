import glovepy
import io, json
import gensim
import time
from gensim.models import word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile

from nltk.stem import WordNetLemmatizer

import codecs

import numpy as np
from glove import Corpus, Glove
import sklearn
from sklearn.cluster import KMeans

from nltk.corpus import stopwords
from nltk.cluster import KMeansClusterer

from collections import Counter

from gensim.models import word2vec
from gensim.models import KeyedVectors
from gensim.utils import lemmatize
from gensim.parsing.porter import PorterStemmer
import logging
import json,io
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
from nltk.corpus import stopwords
from nltk.cluster import KMeansClusterer
import nltk
from sklearn import cluster
from sklearn import metrics
import os
stopWords = set(stopwords.words('english'))
stopWords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '/', '%'])

global nc
nc = []
nc.append(1000)
nc.append(2000)
nc.append(3000)
nc.append(3500)
nc.append(5000)




def read_prepare():
    print('-----read-----')
    input_file = 'articles.json'
    counter = 0
    limit = 1800
    sentences=[]
    stop = []
    myfilename = 'my_stop.txt'
    my_stop = []
    f= open(myfilename,"r")
    with io.open(myfilename, 'r') as stop_file:
        for line in stop_file:
            text = line
            text = text.strip()
            my_stop.append(text)
    print(len(my_stop))
    stop_file.close()
    fname = 'model_output.txt'
    with io.open(input_file, 'r', encoding='utf-8', errors='ignore') as data_file:
        next(data_file)
        for line in data_file:
            if counter  % 1000 == 0:
                print(counter)
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
                    #if w not in my_stop:
                        #stop.append(w)
                    wordsFiltered.append(w)
            sentences.append(wordsFiltered)
    data_file.close()
    ''' 
    print(len(stop))
    c = Counter(stop)
    new_stop = []
    word_counter = {}
    for word in stop:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1
    stop=list(set(stop))
    print('Total words ',len(stop))
    for word in stop:
        if word_counter[word] < 5:
            new_stop.append(word)
    print('Total Stop words ',len(new_stop))
    c = Counter(new_stop)
    #print(len(c))
    print('Total Keep words ',len(stop)-len(c))
            
    popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
    #top_3 = popular_words[:3]
    #print(top_3)
    myfilename = 'my_stop.txt'
    try:
        os.remove(myfilename)
    except OSError:
        pass
    f= open(myfilename,"w+")
    for w in new_stop:
        f.write('%s \n' %w)
    f.close()
    '''

    print('----create Corpus-Glove model----')
    corpus = Corpus()
    corpus.fit(sentences, window=5)
    glove = Glove(no_components=200, learning_rate=0.05)
    print(corpus.matrix)
    glove.fit(corpus.matrix, epochs=10, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')

    print('----Create Glove File----')
    glove = Glove.load('glove.model')
    myfilename = 'glove_file.txt'
    f= open(myfilename,"w+")
    for i in range(len(glove.dictionary)):
        f.write('%s' %list(glove.dictionary)[i])
        for k in glove.word_vectors[i]:
            f.write(' %s' %k)
        f.write('\n')

    

def create_clusters(num_clus):

    print('----Glove to Word2Vec format----')
    glove_file = datapath('glove_file.txt')
    tmp_file = get_tmpfile("test_word2vec.txt")
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    word_vectors = model.wv
    print(len(model.wv.vocab))
    #print(model)

    print('----Clustering----')
    NUM_CLUSTERS=num_clus
    word_vectors = model.wv.syn0
    n_words = word_vectors.shape[0]
    vec_size = word_vectors.shape[1]
    print("#words = {0}, vector size = {1}".format(n_words, vec_size))
    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto')
    idx = kmeans.fit_predict(word_vectors)
    word_centroid_list = list(zip(model.wv.index2word, idx))
    word_centroid_list_sort = sorted(word_centroid_list, key=lambda el: el[1], reverse=False)
    limit=10000
    sample_file = str(limit)+'glove'+str(NUM_CLUSTERS)+'syn.txt'
    file_out = open(sample_file, "w")
    for word_centroid in word_centroid_list_sort:
        line = word_centroid[0] + '  => cluster_'+  str(word_centroid[1])  + '\n'
        file_out.write(line)
    file_out.close()

def create_file():
    glove_file = datapath('glove_file.txt')
    tmp_file = get_tmpfile("test_word2vec.txt")
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    wordnet_lemmatizer = WordNetLemmatizer()
    words = list(model.wv.vocab)
    file_out = open('10000LemmSynGlove.txt', "w")
    for word in words:
        l_word = wordnet_lemmatizer.lemmatize(word,pos='v')
        if l_word != word:
            line = word + ' => ' + l_word + '\n'
            file_out.write(line)
    file_out.close()

def print_time(mytime,method):
    myfilename = 'glove_time.txt'
    f= open(myfilename,"a+")
    f.write('Method: %s  -- ' %method)
    f.write('time: %s ' %mytime)
    f.write('\n')

def glove():
    t1 = time.time()
    read_prepare()
    t = time.time() - t1
    print_time(time.strftime("%H:%M:%S", time.gmtime(t)),str(read_prepare))
    for numbers_clusters in nc:
        t11 = time.time()
        create_clusters(numbers_clusters)
        t = time.time() - t11
        print_time(time.strftime("%H:%M:%S", time.gmtime(t)),str(create_clusters))
    t1 = time.time()
    create_file()
    t = time.time() - t1
    print_time(time.strftime("%H:%M:%S", time.gmtime(t)),str(create_file))


def read_prepare_w2v():
    input_file = 'articles.json'
    print('ok')
    counter = 0
    limit = 10000
    documents=[]
    fname = 'word2vec_model.txt'
    with io.open(input_file, 'r', encoding='utf-8', errors='ignore') as data_file:
        next(data_file)
        model = gensim.models.Word2Vec(iter=1,size=200,min_count=5,window=5,workers =4)
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
                    wordsFiltered.append(w)
            documents.append(wordsFiltered)
    model.build_vocab(documents)
    model.train(documents, total_examples=len(documents), epochs=10)
    model.save(fname)

def create_clusters_w2v(num_cl):
    fname = 'word2vec_model.txt'
    NUM_CLUSTERS = num_cl
    model = gensim.models.Word2Vec.load(fname)
    word_vectors = model.wv
    print(len(model.wv.vocab))
    word_vectors = model.wv.syn0
    n_words = word_vectors.shape[0]
    vec_size = word_vectors.shape[1]
    print("#words = {0}, vector size = {1}".format(n_words, vec_size)) 
    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto')
    idx = kmeans.fit_predict(word_vectors)
    word_centroid_list = list(zip(model.wv.index2word, idx))
    word_centroid_list_sort = sorted(word_centroid_list, key=lambda el: el[1], reverse=False)
    file_out = open('10000w2v'+str(NUM_CLUSTERS)+'syn.txt', "w")
    for word_centroid in word_centroid_list_sort:
        line = word_centroid[0] + '  => cluster_'+  str(word_centroid[1])  + '\n'
        file_out.write(line)
    file_out.close()

def create_file_w2v():
    fname = 'word2vec_model.txt'
    wordnet_lemmatizer = WordNetLemmatizer()
    model = gensim.models.Word2Vec.load(fname)
    words = list(model.wv.vocab)
    file_out = open('10000LemmSyn.txt', "w")
    for word in words:
        l_word = wordnet_lemmatizer.lemmatize(word,pos='v')
        if l_word != word:
            line = word + ' => ' + l_word + '\n'
            file_out.write(line)
    file_out.close()

def print_time_w2v(mytime,method):
    myfilename = 'word2vec_time.txt'
    f= open(myfilename,"a+")
    f.write('Method: %s  -- ' %method)
    f.write('time: %s ' %mytime)
    f.write('\n')

def my_word2vec():
    t1 = time.time()
    read_prepare_w2v()
    t = time.time() - t1
    print_time_w2v(time.strftime("%H:%M:%S", time.gmtime(t)),str(read_prepare_w2v))
    for numbers_clusters in nc:
        t11 = time.time()
        create_clusters_w2v(numbers_clusters)
        t = time.time() - t11
        print_time_w2v(time.strftime("%H:%M:%S", time.gmtime(t)),str(create_clusters_w2v))
    t1 = time.time()
    create_file()
    t = time.time() - t1
    print_time_w2v(time.strftime("%H:%M:%S", time.gmtime(t)),str(create_file_w2v))


