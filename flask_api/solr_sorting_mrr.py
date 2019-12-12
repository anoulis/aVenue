import re
import time
import ast
import numpy as np
import collections
from collections import OrderedDict,Counter
import sys
global dict_of_journals_frequencies,final_rank

# here i make the final rank of top k (10) suggested journals
def get_final_rank(j_names,k):
    #my_list = list(set(j_names))
    my_list = sorted(set(j_names), key=lambda x: j_names.index(x))
    # for k = 10 
    final_rank = my_list[:10]
    return final_rank

# computation of reciprocal rank of one search results
def get_rr(final_rank,old_dic,k):
    try:
        rr = 1 / float(final_rank.index(old_dic["journal"]) + 1 )
    except ValueError:
        rr = 0
    return rr
