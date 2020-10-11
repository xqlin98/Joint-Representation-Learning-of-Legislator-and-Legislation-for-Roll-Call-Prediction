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


if __name__=='__main__':
    time_len = 4
    year = list(range(2005,2018))
    renci_unsaw = []
    renci_saw = []
    member_test = []
    member_train = []
    member_test_train = []
    for time_end in range(2005,2018):
        train, test, member_dict, state_dict, party_dict, member_info, adjacent_matrix = get_all_data(
            time_end-time_len, time_end,load=False,eliminate=True)
        train_member = {}
        test_member = {}
        for leg in train:
            for member in train[leg]:
                if member == 'basic_information':
                    continue
                if member not in train_member:
                    train_member[member] = 1
                else:
                    train_member[member] += 1
        for leg in test:
            for member in test[leg]:
                if member == 'basic_information':
                    continue
                if member not in test_member:
                    test_member[member] = 1
                else:
                    test_member[member] += 1
        un_saw = 0
        saw = 0
        member_test_train_ = 0
        for mem in test_member:
            if mem not in train_member:
                un_saw += test_member[mem]
            else:
                saw += test_member[mem]
                member_test_train_ += 1
        renci_saw.append(saw)
        renci_unsaw.append(un_saw)
        member_test.append(len(test_member))
        member_train.append(len(train_member))
        member_test_train.append(member_test_train_)
        print('{} unsaw:saw={}:{}={}'.format(time_end,un_saw,saw,(un_saw/saw)))
    result = pd.DataFrame({'year':year,'renci_unsaw':renci_unsaw,'renci_saw':renci_saw,'member_train':member_train,'member_test':member_test,'overlap':member_test_train})
    result.to_csv('people_result_house_vote.csv')
