# -*- coding: utf-8 -*
import re, os
import gensim
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet


stopWords = set(stopwords.words('english'))
stopWords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '/', '%', '+', "-"])
wordnet_lemmatizer = WordNetLemmatizer()
name_file = 'my_protwords.txt'

def create_protected_words(mylist):
    '''
    try:
        os.remove(name_file)
    except OSError:
        pass
    '''
    file_out = open(name_file, "a+")
    for word in mylist:
        line = word+'t'
        file_out.write('%s \n' %line)
    file_out.close()

def prepare(mystring, mydocuments):
    mystring = re.sub(r'[^a-zA-Z ]+', '', mystring)
    words = word_tokenize(mystring)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    mydocuments.append(wordsFiltered)
    mystring = " ".join(wordsFiltered)
    return mystring

def create_new_string_title(mystring):
    mystring = re.sub(r'[^a-zA-Z ]+', '', mystring)
    words = word_tokenize(mystring)
    wordsFiltered = []
    for w in words:
        w = str.lower(w)
        if w not in stopWords:
            wordsFiltered.append(w)

    mystring = "t".join(wordsFiltered)

    if mystring != '':
        mystring = mystring + "t"
        #mystring = mystring
        #create_protected_words(wordsFiltered)
    else:
        mystring = 0
    return mystring


def create_new_string_title2(mystring):
    #mystring = re.sub(r'[^a-zA-Z ]+', '', mystring)
    #mystring = re.sub(r'[^a-zA-Z0-9 ]',r'',mystring)
    ms = mystring.replace(' '," ,")
    words = word_tokenize(mystring)
    wordsFiltered = []
    for w in words:
        w = str.lower(w)
        if w not in stopWords:
            wordsFiltered.append(w)
    mystring = "^2 ".join(wordsFiltered)

    if mystring != '':
        mystring = mystring + "^2 "
    else:
        mystring = 0
    #create_protected_words(wordsFiltered)
    return mystring


def create_new_string(mystring):
    mystring = re.sub(r'[^a-zA-Z ]+', '', mystring)
    words = word_tokenize(mystring)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    mystring = ",".join(wordsFiltered)
    return mystring

def create_new_string2(mystring):
    #mystring = re.sub(r'[^a-zA-Z ]+', '', mystring)
    mystring = re.sub(r'[^a-zA-Z0-9 ]',r'',mystring)
    #print(mystring)
    '''
    #print(mystring)
    words = word_tokenize(mystring)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    mystring = " ".join(wordsFiltered)
    '''
    return mystring

def create_new_string22(mystring):
    #mystring = re.sub(r'[^a-zA-Z ]+', '', mystring)
    mystring = re.sub(r'[^a-zA-Z0-9 ]',r'',mystring)
    #mystring = mystring.replace(" ",",")
    #print(mystring)
    '''
    words = word_tokenize(mystring)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    mystring = ",".join(wordsFiltered)
    '''
    return mystring    

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def create_lemmatization(words_string):
    wordsFiltered = []
    '''
    words = gensim.utils.simple_preprocess(words_string)
    tagged_tokens = pos_tag(words)
    for tagged_token in tagged_tokens:
        word = tagged_token[0]
        word_pos = tagged_token[1]
        wntag = get_wordnet_pos(tagged_token[1])
        if wntag!='':
            #print(tagged_token[0],tagged_token[1])
        #else:
            l_word = wordnet_lemmatizer.lemmatize(word,wntag)
            if l_word not in stopWords:
                wordsFiltered.append(l_word)
    my_string = ''
    for word in wordsFiltered:
        my_string = my_string + word + " "
    return my_string

    '''
    words = gensim.utils.simple_preprocess(words_string)
    for w in words:
        l_word = wordnet_lemmatizer.lemmatize(w,pos='v')
        if l_word not in stopWords:
            wordsFiltered.append(l_word)
    my_string = ''
    for word in wordsFiltered:
        my_string = my_string + word + " "
    return my_string





