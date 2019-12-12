import re
import time
import ast
import numpy as np
import collections
from collections import OrderedDict,Counter
import sys

global dict_of_journals_frequencies,final_rank

def get_final_rank(j_names,k):
    #my_list = list(set(j_names))
    my_list = sorted(set(j_names), key=lambda x: j_names.index(x))
    final_rank = my_list[:10]
    return final_rank

def get_final_rank22(top_k,j_names,k):
    global dict_of_journals_frequencies
    dict_of_journals_frequencies = {}
    counter = collections.Counter(j_names)
    j_names = np.array(j_names)
    #print(counter)
    for key, v in counter.most_common(k):
        temp = np.where(j_names == key)[0]
        t_sum = 0
        for x in temp:
            t_sum = t_sum + (1 / (float(x) + 1))
        dict_of_journals_frequencies[key] = t_sum

    final_rank = Counter(dict_of_journals_frequencies)
    final_rank = final_rank.most_common(k)
    return final_rank

def get_rr(final_rank,old_dic,k):
    try:
        rr = 1 / float(final_rank.index(old_dic["journal"]) + 1 )
    except ValueError:
        rr = 0
    return rr





def get_mrr22():

    # sort journals list by the frequency value
    sorted_journals = sorted(dict_of_journals_frequencies, key=dict_of_journals_frequencies.get,
                                         reverse=True)
    sorted_frequencies = sorted(dict_of_journals_frequencies.values(), reverse=True)
    mrr = 0
    if len(sorted_frequencies) == 1:
        mrr = 1
    else:
        for x in range(0, len(sorted_frequencies)):
            if x == 0:
                previous = sorted_frequencies[x]
                number_of_same = 1
                temp_sum = x + 1
                if sorted_frequencies[x] != sorted_frequencies[x+1]:
                    mrr = mrr + (float(1) / (x + 1) )
            else:
                if sorted_frequencies[x] == previous:
                    temp_sum = temp_sum + (x + 1)
                    number_of_same += 1
                    previous = sorted_frequencies[x]
                else:
                    if number_of_same ==1:
                        mrr = mrr + (float(1) / x )
                        if x == len(sorted_frequencies) - 1:
                            mrr = mrr + (float(1) / (x + 1) )
                    else:
                        mrr = mrr + (float(temp_sum) / number_of_same)
                        if x == len(sorted_frequencies) - 1:
                            if sorted_frequencies[x] != previous:
                                mrr = mrr + (float(1) / (x + 1) )
                    previous = sorted_frequencies[x]
                    temp_sum = x + 1
                    number_of_same = 1
    mrr = mrr / len(sorted_frequencies)
    return mrr