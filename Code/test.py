import torch
import torch.nn as nn
import json
from pprint import pprint
import nltk
from nltk.corpus import stopwords
import numpy as np
from trash import get_all_data
import torch.nn.functional as F
from torch import optim
import random
import argparse
import copy
import pandas as pd
import os

if __name__ == '__main__':
    time_len = 4
    with open('legislation.json') as k:
        k = json.load(k)
    for time_end in range(2005, 2018):
        train, test, member_dict, state_dict, party_dict, member_info, adjacent_matrix = get_all_data(
            time_end-time_len, time_end, load=False, eliminate=True)
        count = 0
        with open("{}.txt".format(time_end), "w") as txt:
            print(txt)
        for legislation_name1 in test:
            if 'Official Title as Introduced' not in k[legislation_name1]['title']:
                title_key = 'Official Titles as Introduced'
            else:
                title_key = 'Official Title as Introduced'
            des = test[legislation_name1]['basic_information']['Descrption']
            tit = k[legislation_name1]['title'][title_key]
            if tit[-1] == '.':
                tit = tit[:-1]
            legislation_input = des + \
                ' '+tit + '\n'
            count += int(des==tit)

            with open("{}.txt".format(time_end), "a") as txt:
                txt.write('Title: '+tit)
                txt.write('\n')
                txt.write('Description: '+des)
                txt.write('\n\n')
        print("%d : %.6f num: %d" % (time_end, count/len(test), len(test)))
