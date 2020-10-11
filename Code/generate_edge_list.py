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
    for time_end in range(1997, 2019):
        train, test, member_dict, state_dict, party_dict, member_info, adjacent_matrix = get_all_data(
            time_end-time_len, time_end, load=False, eliminate=False)
        num_member = len(member_dict)
        edge_list = []
        with open("{}_edge_list".format(time_end), "w") as txt:
            print(txt)
        for i in range(num_member):
            for j in range(i+1, num_member):
                if adjacent_matrix[i][j] != 0:
                    edge_list.append([str(i+1),str(j+1),str(adjacent_matrix[i][j])])
        for tmp in edge_list:
            with open("{}_edge_list".format(time_end), "a") as txt:
                txt.write('\t'.join(tmp))
                txt.write('\n')
        
