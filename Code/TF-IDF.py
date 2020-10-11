from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
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


if __name__=='__main__':
    with open('legislation.json') as k:
        k = json.load(k)
    time_len = 4
    for time_end in range(2005, 2018):
        train, test, member_dict, state_dict, party_dict, member_info, adjacent_matrix = get_all_data(
            time_end-time_len, time_end, load=False, eliminate=True)
        tf_idf_model = TfidfVectorizer(binary=False,decode_error='ignore',stop_words='english')
        train_docs = []
        test_docs = []
        for legislation_name1 in train:
            if 'Official Title as Introduced' not in k[legislation_name1]['title']:
                title_key = 'Official Titles as Introduced'
            else:
                title_key = 'Official Title as Introduced'
            des = train[legislation_name1]['basic_information']['Descrption']
            tit = k[legislation_name1]['title'][title_key]
            if tit[-1] == '.':
                tit = tit[:-1]
            train_docs.append(des+tit)
        tf_idf_model.fit(train_docs)
        for legislation_name1 in test:
            if 'Official Title as Introduced' not in k[legislation_name1]['title']:
                title_key = 'Official Titles as Introduced'
            else:
                title_key = 'Official Title as Introduced'
            des = test[legislation_name1]['basic_information']['Descrption']
            tit = k[legislation_name1]['title'][title_key]
            if tit[-1] == '.':
                tit = tit[:-1]
            test_docs.append(des+tit)
        train_legislation_rep = tf_idf_model.transform(train_docs).todense()
        test_legislation_rep = tf_idf_model.transform(test_docs).todense()
        pca = PCA(n_components=32)
        pca.fit(train_legislation_rep)
        train_legislation_rep = pca.transform(train_legislation_rep)
        test_legislation_rep = pca.transform(test_legislation_rep)
        count = 0
        train_legislation_rep_dict = {}
        for leg_name in train:
            train_legislation_rep_dict[leg_name] = train_legislation_rep[count]
            count += 1
        count = 0
        test_legislation_rep_dict = {}
        for leg_name in test:
            test_legislation_rep_dict[leg_name] = test_legislation_rep[count]
            count += 1
        a = 1
        

    vec=cv.fit_transform(['hello world','this is a panda.'])#传入句子组成的list
    arr=vec.toarray()
