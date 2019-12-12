# -*- coding: utf-8 -*-
#import sys
#sys.stdout = open('output.txt','wt')
import json,os

# function to print in file the results of a specific search
def print_results(final_rank,reciprocal_rank,counter1,old_dic,myfilename,k):
    f= open(myfilename,"a+")
    f.write('Num: %d \n' %counter1)
    f.write('Document with id: %s \nTitle: %s \nJournal: %s\n' % (old_dic["id"],old_dic["title"],old_dic["journal"]))
    f.write('Top'+ str(k) + ' journals:\n')
    i = 1
    for k in final_rank:
        f.write("%d " %i)
        f.write(" %s\n" % k)
        i+=1
    f.write('Reciprocal_Rank: %.1f\n\n' %reciprocal_rank)

# function to print the final MRR and time results of a search loop in a specific field
def print_time_MRR(mrr,filename,itime,qtime):
    line = str(mrr)
    with open(filename, 'r+') as f:
        file_data = f.read()
        f.seek(0, 0)
        f.write( 'Index time ' + itime + '\n' + 'Query time ' + qtime + '\n' 'MRR: ' + line.rstrip('\r\n') + '\n' + file_data + '\n')

# function to print the MRR and time results of a method to comparison file of all methods
def print_MRR_Time_comparison(mrr,method,limit,index_time,query_time):
    myfilename = 'Ultimate_comparison' + str(limit)
    f= open(myfilename,"a+")
    f.write('Method: %s  -- ' %method)
    f.write('MRR: %s ' %mrr)
    f.write('Index Time: %s ' %index_time)
    f.write('Query Time: %s ' %query_time)
    f.write('\n')

