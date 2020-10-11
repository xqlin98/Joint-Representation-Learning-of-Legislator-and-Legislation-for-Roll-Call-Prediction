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
from sklearn.metrics import classification_report

class member_gcn(nn.Module):
    def __init__(self,  member_size, state_size,party_size):
        super(member_gcn,self).__init__()
        self.member_embedding = nn.Embedding(member_size, 16)
        self.party_embedding = nn.Embedding(party_size, 8)
        self.state_embeddding = nn.Embedding(state_size, 8)
        self.bn1 = nn.BatchNorm1d(32)
    def forward(self, member, state, party, adjacent_matrix_hat):
        member_embed = self.member_embedding(member)
        party_embed = self.party_embedding(party)
        state_embed = self.state_embeddding(state)
        legislator_embem = torch.cat([member_embed, party_embed, state_embed], dim=1)
        legislator_embem = self.bn1(legislator_embem)
        return legislator_embem


class LSTM_MLP(nn.Module):
    def __init__(self, word_embed_dim, hidden_dim, lstm_out_dim, vocab_size, mask_size, state_size,party_size):
        super(LSTM_MLP, self).__init__()
        self.lstm_out_dim = lstm_out_dim
        self.hidden_dim = hidden_dim
        self.word_embed_dim = word_embed_dim
        # self.w0 = torch.rand(self.lstm_out_dim + 32, gcn0_dim, dtype=torch.float64)
        # self.w1 = torch.rand(gcn0_dim, 3, dtype=torch.float64)
        self.legislation_embedding = nn.Embedding(vocab_size, word_embed_dim)
        self.lstm = nn.LSTM(word_embed_dim, lstm_out_dim)
        self.member_embed = member_gcn(mask_size,state_size,party_size)
        self.mlp1 = nn.Linear(64,32)
        self.mlp2 = nn.Linear(32,3)

    def forward(self, sentence, member_all, state_all, party_all, mask_size, adjacent_matrix):
        # embeding the legislation
        legislation_embed = self.legislation_embedding(sentence)
        # padding
        legislation_embed = torch.cat([legislation_embed, legislation_embed.new_zeros([32 - legislation_embed.size(0),self.word_embed_dim])], 0)
        _, out = self.lstm(legislation_embed.view(32, 1, -1))
        lstm_out = out[0][0]
        lstm_out = lstm_out.expand(mask_size, self.lstm_out_dim)

        # embeding the legislator
        legislator_embem = self.member_embed(member_all, state_all, party_all, adjacent_matrix)

        # normalize
        leg = F.normalize(lstm_out)
        votor = F.normalize(legislator_embem)
        return votor, leg


def sen2index(sen):
    words = nltk.word_tokenize(sen)
    index_ls = []
    for word in words:
        if str.lower(word) in word_dict:
            index_ls.append(word_dict[str.lower(word)])
    if len(index_ls)>32:
        index_ls = index_ls[0:32]
    return torch.tensor(index_ls, dtype=torch.long).to(cuda0)

def get_all_mem_info(member_info):
    member_size = len(member_info)
    member_all = np.array(range(member_size))
    state_all = np.zeros(member_size)
    party_all = np.zeros(member_size)
    for i in range(member_size):
        state_all[i], party_all[i] = member_info[i]
    return member_all, state_all, party_all 
    

def member2index(vote_data, member_size, sample_num=1000):
    member_pos,member_neg = [],[]
    member_all, state_all, party_all = get_all_mem_info(member_info)
    vote_result_dict = {0:[],1:[],2:[]}
    # get the vote data
    for vote in vote_data:
        if vote == 'basic_information' or vote_data[vote]['Vote'] not in vote_dict:
            continue
        record = vote_data[vote]
        vote_result_dict[vote_dict[record['Vote']]].append(member_dict[vote])
        state_all[member_dict[vote]] = state_dict[record['State']]
        party_all[member_dict[vote]] = party_dict[record['Party']]
    # sample from postive and negtive
    vote_kind = [tmp for tmp in vote_result_dict if vote_result_dict[tmp]]
    if len(vote_kind)<2:
        return None,None,None,None,None
    
    for i in range(sample_num):
        pos_vote,neg_vote = np.sort(np.random.choice(vote_kind,2,replace=False))
        pos_member, neg_member = np.random.choice(vote_result_dict[pos_vote],1), np.random.choice(vote_result_dict[neg_vote],1)
        member_pos.append(pos_member)
        member_neg.append(neg_member)

    return torch.tensor(member_all, dtype=torch.long).to(cuda0), torch.tensor(state_all, dtype=torch.long).to(cuda0), \
        torch.tensor(party_all, dtype=torch.long).to(cuda0), torch.tensor(member_pos, dtype=torch.long).to(cuda0), \
        torch.tensor(member_neg, dtype=torch.long).to(cuda0)


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
    return torch.tensor(member_all, dtype=torch.long).to(cuda0), torch.tensor(state_all, dtype=torch.long).to(cuda0), \
        torch.tensor(party_all, dtype=torch.long).to(cuda0), vote_result, \
        torch.tensor(member_mask, dtype=torch.long), vote_num

def Learning_rep(args):
    # define model
    lstmgcn_model = LSTM_MLP(word_embeddim, lstm_outdim, lstm_outdim,
                                len(word_dict), len(member_dict), len(state_dict), len(party_dict)).to(cuda0)
    pdist = nn.PairwiseDistance(2)
    optimizer = optim.Adam(lstmgcn_model.parameters(), lr=args.lr)

    # some set up for training and split train and val
    train_name_list = [name for name in train]
    random.shuffle(train_name_list)
    val_name_list = train_name_list[:len(train_name_list)//5]
    train_name_list = train_name_list[len(train_name_list)//5:]
    
    count = 0
    loss_avg = 0
    best_acc = 0
    
    early_stop_count = 0

    # begin training
    for epoch in range(args.epochs):
        print('epoch' + str(epoch))
        for legislation_name in train_name_list:
            count += 1
            
            # evaluations
            if count % args.eval == 1:
                print('Loss of Train set is %.4f' % (loss_avg/200.0))
                loss_avg = 0
                total = 0
                right = 0
                with torch.no_grad():
                    real_freq = {0: 0, 1: 0, 2: 0}
                    predict_freq = {0: 0, 1: 0, 2: 0}
                    for legislation_name1 in val_name_list:
                        if 'Official Title as Introduced' not in k[legislation_name1]['title']:
                            title_key = 'Official Titles as Introduced'
                        else:
                            title_key = 'Official Title as Introduced'
                        legislation_input = sen2index(
                            train[legislation_name1]['basic_information']['Descrption']+' '+k[legislation_name1]['title'][title_key])
                        if legislation_input.size()[0] == 0:
                            continue
                        
                        member_input, state_input, party_input, gt_result, member_mask, vote_num  = member2index_eval(train[legislation_name1],len(member_dict))
                        if len(member_input) == 0:
                            continue

                        # get the representation
                        member_rep, leg_rep = lstmgcn_model(legislation_input, member_input, state_input, party_input,len(member_mask),adjacent_matrix)
                        member_rep = member_rep[member_mask,:]
                        
                        # prediction
                        pred_result = np.zeros(len(member_input))
                        pair_distance = pdist(leg_rep, member_rep)
                        _,member_idx = torch.sort(pair_distance)
                        sorted_member = member_mask[member_idx]
                        pred_result[sorted_member[0:vote_num[0]]]=0
                        pred_result[sorted_member[vote_num[0]:sum(vote_num[0:2])]]=1
                        pred_result[sorted_member[sum(vote_num[0:2]):sum(vote_num[0:3])]]=2
                        
                        # mask
                        pred_result = pred_result[member_mask]
                        gt_result = gt_result[member_mask]
                        # compute the accuracy
                        right += sum(pred_result == gt_result)
                        total += len(member_mask)
                        for i in range(len(pred_result)):
                            predict_freq[pred_result[i]] += 1
                            real_freq[gt_result[i]] += 1
                # save the best model and early stop
                if best_acc < (right / total):
                    best_acc = (right / total)
                    best_model = LSTM_MLP(word_embeddim, lstm_outdim, lstm_outdim,
                                len(word_dict), len(member_dict), len(state_dict), len(party_dict)).to(cuda0)
                    best_model.load_state_dict(copy.deepcopy(lstmgcn_model.state_dict()))
                    early_stop_count = 0
                early_stop_count += 1
                if early_stop_count >=args.early_stop:
                    break
                # the frequence of different classes
                print('----- predict_freq -----')
                print(predict_freq)
                print('----- real_freq -----')
                print(real_freq)
                print('LSTM_GCN acc:' + str(right / total) + '\t'+ '\t' + 'epoch:' + str(epoch) +
                       '\t\t' + 'Data: '+str(time_end-time_len)+'-'+str(time_end)+'\t'+'Best acc:'+str(best_acc))
            
            # get the data prepared
            if 'Official Title as Introduced' not in k[legislation_name]['title']:
                title_key = 'Official Titles as Introduced'
            else:
                title_key = 'Official Title as Introduced'
            legislation_input = sen2index(
                train[legislation_name]['basic_information']['Descrption']+' '+k[legislation_name]['title'][title_key])
            if legislation_input.size()[0] == 0:
                continue
            member_all, state_all, party_all, member_pos, member_neg \
                 = member2index(train[legislation_name],len(member_dict))

            if member_all is None:
                continue
            
            member_rep, leg_rep = lstmgcn_model(
                legislation_input, member_all, state_all, party_all, len(member_pos), adjacent_matrix)
            
            # get the representation
            member_rep_pos, member_rep_neg = member_rep[member_pos,:], member_rep[member_neg,:]
            
            # compute the loss for the representation
            per_pair_loss = pdist(leg_rep, member_rep_pos) - pdist(leg_rep, member_rep_neg)
            per_pair_loss = F.relu(per_pair_loss+10.0)
            loss = per_pair_loss.mean()
            loss_avg += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if early_stop_count >=args.early_stop:
            break
    return best_acc, best_model


def prediction_on_test(args,rep_model,leg_dict,predicted_vote_rate, gt_vote_rate_all):
    rep_model.eval()
    pdist = nn.PairwiseDistance(2)
    # prediction on test set
    total = 0
    right = 0
    right_gt = 0
    gt_result_all = []
    pred_result_all = []
    pred_result_gt_all = []
    with torch.no_grad():
        real_freq = {0: 0, 1: 0, 2: 0}
        predict_freq = {0: 0, 1: 0, 2: 0}
        for legislation_name1 in test:
            if 'Official Title as Introduced' not in k[legislation_name1]['title']:
                title_key = 'Official Titles as Introduced'
            else:
                title_key = 'Official Title as Introduced'
            legislation_input = sen2index(
                test[legislation_name1]['basic_information']['Descrption']+' '+k[legislation_name1]['title'][title_key])
            if legislation_input.size()[0] == 0:
                continue
            
            member_input, state_input, party_input, gt_result, member_mask, _  = member2index_eval(test[legislation_name1],len(member_dict))
            if len(member_input) == 0:
                continue

            # get the predicted vote rate
            vote_num = len(member_mask)
            vote_rate = predicted_vote_rate[leg_dict[legislation_name1],:].tolist()
            pred_vote_num = [int(tmp*vote_num) for tmp in vote_rate]
            if sum(pred_vote_num) != vote_num:
                pred_vote_num[0] += (vote_num-sum(pred_vote_num))
            
            # get the ground true vote rate
            gt_vote_rate = gt_vote_rate_all[leg_dict[legislation_name1],:].tolist()
            gt_vote_num = [int(tmp*vote_num) for tmp in gt_vote_rate]
            if sum(gt_vote_num) != vote_num:
                gt_vote_num[0] += (vote_num-sum(gt_vote_num))
                
            # get the representation
            member_rep, leg_rep = rep_model(legislation_input, member_input, state_input, party_input,len(member_mask),adjacent_matrix)
            member_rep = member_rep[member_mask,:]
            
            # prediction
            pred_result = np.zeros(len(member_input))
            pair_distance = pdist(leg_rep, member_rep)
            _,member_idx = torch.sort(pair_distance)
            sorted_member = member_mask[member_idx]
            pred_result[sorted_member[0:pred_vote_num[0]]]=0
            pred_result[sorted_member[pred_vote_num[0]:sum(pred_vote_num[0:2])]]=1
            pred_result[sorted_member[sum(pred_vote_num[0:2]):sum(pred_vote_num[0:3])]]=2

            # prediction with ground true vote rate
            pred_result_gt = np.zeros(len(member_input))
            pred_result_gt[sorted_member[0:gt_vote_num[0]]]=0
            pred_result_gt[sorted_member[gt_vote_num[0]:sum(gt_vote_num[0:2])]]=1
            pred_result_gt[sorted_member[sum(gt_vote_num[0:2]):sum(gt_vote_num[0:3])]]=2
            
            # mask 
            pred_result_gt = pred_result_gt[member_mask]
            pred_result = pred_result[member_mask]
            gt_result = gt_result[member_mask]
            
            # compute the accuracy
            right += sum(pred_result == gt_result)
            right_gt += sum(pred_result_gt == gt_result)
            total += len(member_mask)
            for i in range(len(pred_result)):
                predict_freq[pred_result[i]] += 1
                real_freq[gt_result[i]] += 1
            gt_result_all.append(np.squeeze(gt_result))
            pred_result_all.append(np.squeeze(pred_result))
            pred_result_gt_all.append(np.squeeze(pred_result_gt))


    # compute the recall and f1 score
    gt_result_all = np.concatenate(gt_result_all)
    pred_result_all = np.concatenate(pred_result_all)
    pred_result_gt_all = np.concatenate(pred_result_gt_all)
    target_names = ['Yes','Not vote','No']
    metrics = classification_report(gt_result_all, pred_result_all, target_names=target_names,output_dict=True)
    wei_recall=metrics['weighted avg']['recall']
    micro_recall = metrics['micro avg']['recall']
    wei_f1 = metrics['weighted avg']['f1-score']
    micro_f1 = metrics['micro avg']['f1-score']
    
    metrics1 = classification_report(gt_result_all, pred_result_gt_all, target_names=target_names,output_dict=True)
    wei_recall_gt=metrics1['weighted avg']['recall']
    micro_recall_gt = metrics1['micro avg']['recall']
    wei_f1_gt = metrics1['weighted avg']['f1-score']
    micro_f1_gt = metrics1['micro avg']['f1-score']
    
    # the frequence of different classes
    print('----- predict_freq -----')
    print(predict_freq)
    print('----- real_freq -----')
    print(real_freq)
    print('Final LSTM_GCN_tri_vote_rate acc:' + str(right / total) +
            '\t\t' + 'Data: '+str(time_end-time_len)+'-'+str(time_end)+'\t'+'GT vote rate acc:'+str(right_gt/total))
    return right / total, right_gt/total, wei_recall, micro_recall, wei_f1, micro_f1, wei_recall_gt, micro_recall_gt, wei_f1_gt, micro_f1_gt


if __name__ == '__main__':
    # important parameters: semantic, result, lstm
    parser = argparse.ArgumentParser(
        description='HyperParameters for String Embedding')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--word_dim', type=int, default=32,
                        help='word dimension')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--version', type=int, default=1,
                    help='version info')
    parser.add_argument('--eval', type=int, default=100,
                    help='evaluation frequency')
    parser.add_argument('--load', type=int, default=0,
                    help='load trained model')
    parser.add_argument('--time_start', type=int, default=2005,
                    help='whether to check result')
    parser.add_argument('--time_end', type=int, default=2017,
                    help='whether to check result')
    parser.add_argument('--time_len', type=int, default=4,
                    help='whether to check result')
    parser.add_argument('--cuda', type=int, default=0,
                    help='whether to check result')
    parser.add_argument('--lstm', type=int, default=1,
                    help='whether to use lstm in vote rate prediction')
    parser.add_argument('--semantic', type=int, default=1,
                    help='whether to use semantic gcn')
    parser.add_argument('--result', type=str, default='lstm_gcn_tri_pred_vote',
                    help='define the name for result file')
    parser.add_argument('--co_ocur', type=int, default=0,
                    help='whether to use co-ocurence adj matrix to build semantic gcn')
    parser.add_argument('--cat', type=int, default=1,
                    help='whether to cat semantic GCN output and lstm output')
    parser.add_argument('--early_stop', type=int, default=15,
                    help='early stop count')
    args = parser.parse_args()
    cuda0 = torch.device('cuda:{}'.format(args.cuda))
    # cuda0 = torch.device('cpu')
    # parameters
    word_embeddim = 100
    lstm_outdim = args.word_dim
    word_dict_len = 20000
    time_end = args.time_end
    time_len = args.time_len

    # load data
    with open('word_dict.json') as f:
        word_dict_all = json.load(f)
    word_dict = {}
    all_word = list(word_dict_all)
    for i in range(word_dict_len):
        word_dict[all_word[i]] = word_dict_all[all_word[i]]
    with open('legislation.json') as k:
        k = json.load(k)
    with open('house_vote.json') as f:
        house_vote = json.load(f)
    vote_dict = {'Yea': 0, 'Aye': 0, 'Nay': 2,
                 'No': 2, 'Not Voting': 1, 'Present': 1}
    acc = []
    acc_val = []
    acc_gt_vote_rate = []
    wei_recall_all, micro_recall_all, wei_f1_all, micro_f1_all, wei_recall_gt_all, micro_recall_gt_all, wei_f1_gt_all, micro_f1_gt_all = [],[],[],[],[],[],[],[]

    for time_end in range(args.time_start,args.time_end+1):
        # get the data this year
        train, test, member_dict, state_dict, party_dict, member_info, adjacent_matrix = get_all_data(
            time_end-time_len, time_end,load=False,eliminate=True)
        
        # get the menber in training data
        train_member = []
        for leg in train:
            for member in train[leg]:
                if member == 'basic_information':
                    continue
                if member not in train_member:
                    train_member.append(member)
        
        adjacent_matrix = torch.tensor(adjacent_matrix, dtype=torch.float32).to(cuda0)
        
        # get the predicted vote rate
        if args.lstm:
            from semantic_gcn_lstm import main_semantic_gcn
            predicted_vote_rate, leg_dict, gt_vote_rate = main_semantic_gcn(train,test,word_dict,args)
        else:
            from semantic_gcn import main_semantic_gcn
            predicted_vote_rate, leg_dict, gt_vote_rate = main_semantic_gcn(train,test,args)
        
        # Representation learning model
        acc_val_, learning_model = Learning_rep(args)
        
        # make prediction on test set
        acc_, acc_gt_vt, wei_recall, micro_recall, wei_f1, micro_f1, wei_recall_gt, micro_recall_gt, wei_f1_gt, micro_f1_gt = prediction_on_test(args,learning_model,leg_dict,predicted_vote_rate, gt_vote_rate)
        wei_recall_all.append(wei_recall); micro_recall_all.append(micro_recall); wei_f1_all.append(wei_f1); micro_f1_all.append(micro_f1); wei_recall_gt_all.append(wei_recall_gt); micro_recall_gt_all.append(micro_recall_gt); wei_f1_gt_all.append(wei_f1_gt); micro_f1_gt_all.append(micro_f1_gt)

        acc.append(acc_) # accuracy on test set
        acc_gt_vote_rate.append(acc_gt_vt) # with telling the ground true vote rate
        acc_val.append(acc_val_) # accuracy on validation set with known vote rate
        print('Runing acc: Test acc %.6f, Val acc %.6f' % (np.mean(acc), np.mean(acc_val)))
    acc = np.array(acc)
    acc_val = np.array(acc_val)
    acc_gt_vote_rate = np.array(acc_gt_vote_rate)
    result1 = pd.DataFrame({'wei_recall':wei_recall_all,'micro_recall':micro_recall_all, 'wei_f1':wei_f1_all, 'micro_f1':micro_f1_all,'wei_recall_gt':wei_recall_gt_all,'micro_recall_gt':micro_recall_gt_all, 'wei_f1_gt':wei_f1_gt_all, 'micro_f1_gt':micro_f1_gt_all})
    result = pd.DataFrame({'test acc':acc,'val acc':acc_val,'test acc with gt vote rate':acc_gt_vote_rate})
    result.to_csv('{}.csv'.format(args.result))
    result1.to_csv('{}_recall_f1.csv'.format(args.result))
    print('{} Final test acc: %.6f, val acc: %.6f, test acc with gt vote rate: %.6f'.format(args.result) % (np.mean(acc),np.mean(acc_val),np.mean(acc_gt_vote_rate)))
