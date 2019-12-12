# -*- coding: utf-8 -*-
import re
import time
import ast
import numpy as np
import indexing
import search
import collections
from collections import OrderedDict,Counter
import search
from collections import OrderedDict
import sys
articles=indexing.articles
dict_of_journals_frequencies = {}
def get_final_rank(top_k, k):
    journals_names = []
    # keep jpurnals names so in an temp easy-handled array
    for key, v in top_k:
        journals_names.append(articles["journal"][key])

    counter = collections.Counter(journals_names)

    # format to np array
    journals_names = np.array(journals_names)

    # creat a dict to store unique entry of journal with custom frequency value

    for key, v in counter.most_common(k):
        temp = np.where(journals_names == key)[0]
        t_sum = 0
        for x in temp:
            t_sum = t_sum + (1 / (float(x) + 1))
        dict_of_journals_frequencies[key] = t_sum


    # here i make the whole mrr computation with the solution of tie-frequencies cases
    final_rank = Counter(dict_of_journals_frequencies)
    final_rank = final_rank.most_common(k)

    return final_rank


def get_mrr():

    # sort journals list by the frequency value
    sorted_journals = sorted(dict_of_journals_frequencies, key=dict_of_journals_frequencies.get,
                                         reverse=True)
    sorted_frequencies = sorted(dict_of_journals_frequencies.values(), reverse=True)
    previous = sorted_frequencies[0]
    number_of_same = 1
    temp_sum = 1
    mrr = 0
    for x in range(1, len(sorted_frequencies)):
        if sorted_frequencies[x] == previous:
            temp_sum = temp_sum + x + 1
            number_of_same = number_of_same + 1
            previous = sorted_frequencies[x]
        else:
            mrr = mrr + number_of_same * (1 / (float(temp_sum) / number_of_same))
            previous = sorted_frequencies[x]
            temp_sum = x + 1
            number_of_same = 1

        if x + 1 == len(sorted_frequencies):
            mrr = mrr + number_of_same * (1 / (float(temp_sum) / number_of_same))


    mrr = mrr / len(sorted_frequencies)
    return mrr

