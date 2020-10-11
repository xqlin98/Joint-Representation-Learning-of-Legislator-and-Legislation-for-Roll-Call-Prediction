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


def get_all_mem_info(member_info):
    member_size = len(member_info)
    member_all = np.array(range(member_size))
    state_all = np.zeros(member_size)
    party_all = np.zeros(member_size)
    for i in range(member_size):
        state_all[i], party_all[i] = member_info[i]
    return member_all, state_all, party_all 
    

def member2index_eval(vote_data, member_size):
    member_all, state_all, party_all = get_all_mem_info(member_info)
    vote_result = np.zeros(member_size)
    vote_result_dict = {0:[],1:[],2:[]}
    member_mask = []
    # get the vote data
    for vote in vote_data:
        if vote == 'basic_information' or vote_data[vote]['Vote'] not in vote_dict:
            continue
        record = vote_data[vote]
        vote_result_dict[vote_dict[record['Vote']]].append(member_dict[vote])
        state_all[member_dict[vote]] = state_dict[record['State']]
        party_all[member_dict[vote]] = party_dict[record['Party']]
        vote_result[member_dict[vote]] = vote_dict[record['Vote']]
        if vote in train_member:
            member_mask.append(member_dict[vote])
    vote_num = [len(vote_result_dict[tmp]) for tmp in vote_result_dict]
    vote_rate = [tmp/sum(vote_num) for tmp in vote_num]
    return vote_rate

if __name__ == '__main__':
    vote_dict = {'Yea': 0, 'Aye': 0, 'Nay': 2,
                 'No': 2, 'Not Voting': 1, 'Present': 1}
    time_len = 4
    with open('legislation.json') as k:
        k = json.load(k)
    for time_end in range(2005, 2018):
        train, test, member_dict, state_dict, party_dict, member_info, adjacent_matrix = get_all_data(
            time_end-time_len, time_end, load=False, eliminate=True)

        # get the menber in training data
        train_member = []
        for leg in train:
            for member in train[leg]:
                if member == 'basic_information':
                    continue
                if member not in train_member:
                    train_member.append(member)
        vote_rate = []
        for legislation_name1 in test:
            if 'Official Title as Introduced' not in k[legislation_name1]['title']:
                title_key = 'Official Titles as Introduced'
            else:
                title_key = 'Official Title as Introduced'
            vote_rate.append(member2index_eval(test[legislation_name1],len(member_dict)))
        vote_rate = np.array(vote_rate)
        a,b,c = np.mean(vote_rate,axis=0)
        print('---- Year {} ----'.format(time_end))
        print('avg vote rate: %.6f %.6f %.6f' % (a,b,c))
        print('var of vote rate %.6f %.6f %.6f' % (np.std(vote_rate[:,0]),np.std(vote_rate[:,1]),np.std(vote_rate[:,2])))
        vote_rate_table = pd.DataFrame(vote_rate)
        vote_rate_table.to_csv('{}_vote_rate.csv'.format(time_end))
