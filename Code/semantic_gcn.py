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
import time
import argparse
import networkx as nx
import re
import scipy.sparse
from tqdm import tqdm
import pickle
import copy


stws = stopwords.words('english')
vote_dict = {'Yea': 0, 'Aye': 0, 'Nay': 2,
             'No': 2, 'Not Voting': 1, 'Present': 1}
cuda0 = 'cuda'

def save_obj(obj, name):
    with open('obj' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def cosSimilarity(vector1, vector2):
    cs = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cs


def textRank(text, windows_size=4):
    punctuation = ['!', ',', '.', '?']
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    temp_list = text.split()
    word_list = [word for word in temp_list if word not in stws]
    word_list_unique = list(set(word_list))
    graph_matrix = np.zeros((len(word_list_unique), len(word_list_unique)))
    for i in range(len(word_list)):
        for j in range(windows_size):
            if i - j - 1 >= 0:
                graph_matrix[word_list_unique.index(
                    word_list[i])][word_list_unique.index(word_list[i - j - 1])] = 1
                graph_matrix[word_list_unique.index(
                    word_list[i - j - 1])][word_list_unique.index(word_list[i])] = 1
            if i + j + 1 <= len(word_list) - 1:
                graph_matrix[word_list_unique.index(
                    word_list[i])][word_list_unique.index(word_list[i + j + 1])] = 1
                graph_matrix[word_list_unique.index(
                    word_list[i + j + 1])][word_list_unique.index(word_list[i])] = 1

    graph_matrix = scipy.sparse.csr_matrix(graph_matrix)

    nx_graph = nx.from_scipy_sparse_matrix(graph_matrix)
    scores = nx.pagerank(nx_graph)
    return [word[1][0] for word in
            sorted([[i, [s, scores[i]]] for i, s in enumerate(word_list_unique)], key=lambda x: -x[1][1])]


def cal_vote_rate(vote_data):
    vote_result_dict = {0: 0, 1: 0, 2: 0}
    # get the vote data
    for vote in vote_data:
        if vote == 'basic_information' or vote_data[vote]['Vote'] not in vote_dict:
            continue
        record = vote_data[vote]
        vote_result_dict[vote_dict[record['Vote']]] += 1
    vote_num = sum([vote_result_dict[tmp] for tmp in vote_result_dict])
    vote_rate = [vote_result_dict[tmp]/vote_num for tmp in vote_result_dict]
    return vote_rate


def cal_semantic_adj(legislation_dict, load=False):
    embeddings_index = load_obj('glove')
    with open('legislation.json') as k:
        k = json.load(k)
    adj_mtx = np.zeros(
        (len(legislation_dict), len(legislation_dict)), dtype=int)
    legislation_list = list(legislation_dict.keys())
    legislation_index_dict = {
        k: legislation_list.index(k) for k in legislation_list}

    if load:
        adj_mtx = np.load('semantic_adj.npy')
        legislation_rep = np.load('legislation_rep.npy')
        leg_vote_rate = np.load('leg_vote_rate.npy')
        return adj_mtx, legislation_index_dict, legislation_rep, leg_vote_rate

    legislation_index_dict_reverse = {
        v: k for k, v in legislation_index_dict.items()}
    legislation_rep = np.zeros([len(legislation_list), 100])
    leg_vote_rate = np.zeros([len(legislation_list), 3])
    for i in range(len(legislation_list)):
        if 'Official Title as Introduced' not in k[legislation_list[i]]['title']:
            title_key = 'Official Titles as Introduced'
        else:
            title_key = 'Official Title as Introduced'
        legislation_text = legislation_dict[legislation_list[i]]['basic_information']['Descrption'] + \
            k[legislation_list[i]]['title'][title_key]
        keyword_list = textRank(legislation_text)
        if len(keyword_list) > 10:
            keyword_list = keyword_list[:10]
        text_vector = np.zeros(100)
        count = 0
        for kw in keyword_list:
            if kw in embeddings_index:
                text_vector += embeddings_index[kw]
                count += 1
        text_vector = text_vector / count
        legislation_index_dict_reverse[i] = text_vector
        legislation_rep[i] = text_vector
        leg_vote_rate[i] = cal_vote_rate(legislation_dict[legislation_list[i]])
    for i in range(len(legislation_dict)):
        for j in range(len(legislation_dict)):
            adj_mtx[i][j] = cosSimilarity(
                legislation_index_dict_reverse[i], legislation_index_dict_reverse[j])
    np.save('semantic_adj', adj_mtx)
    np.save('legislation_rep', legislation_rep)
    np.save('leg_vote_rate', leg_vote_rate)
    return adj_mtx, legislation_index_dict, legislation_rep, leg_vote_rate


def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0)).to(cuda0)
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class GCN(nn.Module):
    '''
    Z = AXW
    '''

    def __init__(self, dim_in, dim_hidden, dim_out):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=False)
        #self.fc1 = nn.Linear(dim_in ,dim_out,bias=False)

    def forward(self, A, X):
        '''
        计算俩层gcn
        '''
        self.A = A
        X = F.relu(self.fc1(self.A.mm(X)))
        return self.fc2(self.A.mm(X))


class Semantic_GCN(nn.Module):
    def __init__(self):
        super(Semantic_GCN, self).__init__()
        self.bn1 = nn.BatchNorm1d(100)
        self.gcn = GCN(100, 32, 32)
        self.mlp1 = nn.Linear(32, 16)
        self.mlp2 = nn.Linear(16, 3)
        self.soft_max = nn.Softmax()

    def forward(self, legislation_rep, adj_matrix):
        x = self.bn1(legislation_rep)
        x = self.gcn(adj_matrix, x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.normalize(x)

        return x


def get_mask(batch, leg_idx_dict):
    mask = []
    for tmp in batch:
        mask.append(leg_idx_dict[tmp])
    return np.array(mask)


def main_semantic_gcn(train, test, args):
    global cuda0
    cuda0 = torch.device('cuda:{}'.format(args.cuda))
    all_data = {}
    all_data.update(train)
    all_data.update(test)
    
    # calculate 
    semantic_adj, leg_idx_dict, leg_rep, leg_vote_rate = cal_semantic_adj(all_data, load=False)
    leg_rep = torch.tensor(leg_rep, dtype=torch.float32).to(cuda0)
    semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32).to(cuda0)
    leg_vote_rate = torch.tensor(leg_vote_rate, dtype=torch.float32).to(cuda0)
    # parameters
    batch_size = 100
    steps = 10000

    # construct model
    semantic_gcn = Semantic_GCN().to(cuda0)
    mse = nn.MSELoss()
    optimizer = optim.Adam(semantic_gcn.parameters(), lr=0.001)
    all_key = list(train)
    random.shuffle(all_key)
    val_key = all_key[:len(all_key)//5]
    train_key = all_key[len(all_key)//5:]

    early_stop_count = 0
    min_val_loss = 100000

    for step in range(steps):
        train_batch = np.random.choice(train_key, batch_size, replace=False)
        y_mask = get_mask(train_batch, leg_idx_dict)

        # binary mask tensor
        y_mask_b = np.zeros(len(leg_idx_dict))
        y_mask_b[y_mask] = 1.0
        y_mask_b = torch.tensor(y_mask_b, dtype=torch.float32).to(cuda0)

        predicted_rate = semantic_gcn(leg_rep, semantic_adj)
        loss = torch.mean(mse(predicted_rate, leg_vote_rate).mul(y_mask_b))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step % 10) == 0:

            y_mask = get_mask(val_key, leg_idx_dict)
            # binary mask tensor
            y_mask_b = np.zeros(len(leg_idx_dict))
            y_mask_b = np.zeros(len(leg_idx_dict))
            y_mask_b[y_mask] = 1.0
            y_mask_b = torch.tensor(y_mask_b, dtype=torch.float32).to(cuda0)

            predicted_rate = semantic_gcn(leg_rep, semantic_adj)
            loss = torch.mean(mse(predicted_rate, leg_vote_rate).mul(y_mask_b))
            print('Vote rate validation loss:%.6f' % loss.item())
            if min_val_loss > loss.item():
                min_val_loss = loss.item()
                early_stop_count = 0
                best_model = predicted_rate.cpu().detach().numpy()

            early_stop_count += 1
            if early_stop_count >= 20:
                break
    return best_model, leg_idx_dict, leg_vote_rate.cpu().detach().numpy()

