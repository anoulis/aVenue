# -*- coding: utf-8 -*
import re, os
import gensim
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# stopword list creation, init the lemmatizer
stopWords = set(stopwords.words('english'))
stopWords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '/', '%', '+', "-"])
wordnet_lemmatizer = WordNetLemmatizer()

# funtion to create a file for protected words
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

# function to prepare string for indexing, instead of parsing
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

# function to prepare string for querying, instead of parsing
def create_new_string(mystring):
    mystring = re.sub(r'[^a-zA-Z ]+', '', mystring)
    words = word_tokenize(mystring)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    mystring = ",".join(wordsFiltered)
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

# lemmatization function for a given string. Best way when behave any word as verb,
# second we examination of every case
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





