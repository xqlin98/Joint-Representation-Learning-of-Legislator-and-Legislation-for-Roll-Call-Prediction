from pprint import pprint
import json
import nltk
from nltk.corpus import stopwords
import numpy as np
import pickle
from tqdm import tqdm

def save_obj(obj, name):
    with open('obj'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_word_dict():
    with open('legislation.json') as f:
        f = json.load(f)
    text = []
    for i in f:
        if 'title' in f[i]:
            for j in f[i]['title']:
                text.append(f[i]['title'][j])
        if 'summary' in f[i]:
            for j in f[i]['summary']:
                text.append(j)
    text = ' '.join(text)
    print(len(text))
    print('tokenize...')
    text = nltk.word_tokenize(text)
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '/', '-',
                            '<', '>']
    text_list = [word.lower() for word in text if word not in english_punctuations]
    # 去掉停用词
    stops = set(stopwords.words("english"))
    print('textlist...')
    text_list = [word for word in text_list if word not in stops]
    print('freq...')
    freq_dist = nltk.FreqDist(text_list)
    d1 = sorted(freq_dist.items(), key=lambda d: d[1], reverse=True)
    word_dict = {}
    for i in range(len(d1)):
        word_dict[d1[i][0]] = i + 1
    with open('word_dict.json', 'w') as json_file:
        json_file.write(json.dumps(word_dict))


# def adjacent_matrix():
# 	adjacent_dict = {}
# 	with open('house_vote.json') as f:
# 		f = json.load(f)
# 	with open('member_dic.json') as g:
# 		member_dic = json.load(g)
# 	member_dic_reverse = {v[1]: k for k, v in member_dic.items()}
# 	for i in f:
# 		if f[i]['cosponsors']:
# 			cosponsors_list = []
# 			for index in range(len(f[i]['cosponsors'])):
# 				if f[i]['cosponsors'][index][1] not in adjacent_dict:
# 					adjacent_dict[f[i]['cosponsors'][index][1]] = []
# 				else:
# 					continue
# 	pass



def get_all_data(time_limit_begin, time_limit_end, load=False,graph=False,eliminate=False):
    if load:
        train_vote = load_obj('train_vote') 
        test_vote = load_obj('test_vote')
        member_dict = load_obj('member_dict')
        state_dict = load_obj('state_dict')
        party_dict = load_obj('party_dict')
        adjacent_matrix = np.load('adj_matrix.npy')
        edge_list = load_obj('edge_list')
        if graph:
            return train_vote, test_vote, member_dict, state_dict, party_dict, adjacent_matrix, edge_list
        return train_vote, test_vote, member_dict, state_dict, party_dict, adjacent_matrix
    with open('house_vote.json') as f:
        f = json.load(f)
    with open('legislation.json') as legislation:
        legislation = json.load(legislation)
    with open('member_dic.json') as member_dic:
        member_dic = json.load(member_dic)
    member_dic_reverse = {v[0][1]: k for k, v in member_dic.items()}
    legislation_list = []
    member_dict = {} # member from the house vote data
    state_dict = {}
    party_dict = {}
    train_vote = {}
    test_vote = {}
    member_info = {}
    for i in f:
        year = eval(f[i]['basic_information']['Date'].split('-')[-1])
        if year >= time_limit_begin and year < time_limit_end:
            legislation_list.append('.'.join(i.split(' ')))
            if '.'.join(i.split(' ')) not in legislation or len(legislation['.'.join(i.split(' '))]) == 0:
                    continue
            train_vote['.'.join(i.split(' '))] = f[i]
            for member in f[i]:
                if member == 'basic_information':
                    continue
                member_dict[member] = 0
                state_dict[f[i][member]['State']] = 0
                party_dict[f[i][member]['Party']] = 0
                member_info[member] = [f[i][member]['State'],f[i][member]['Party']]

        elif year == time_limit_end:
            #legislation_list.append('.'.join(i.split(' ')))
            if '.'.join(i.split(' ')) not in legislation or len(legislation['.'.join(i.split(' '))]) == 0:
                    continue
            test_vote['.'.join(i.split(' '))] = f[i]
            for member in f[i]:
                if member == 'basic_information':
                    continue
                member_dict[member] = 0
                state_dict[f[i][member]['State']] = 0
                party_dict[f[i][member]['Party']] = 0
                member_info[member] = [f[i][member]['State'],f[i][member]['Party']]

    member_list = list(member_dict)
    state_list = list(state_dict)
    party_list = list(party_dict)

    for i in member_dict:
        member_dict[i] = member_list.index(i)
    for i in state_dict:
        state_dict[i] = state_list.index(i)
    for i in party_dict:
        party_dict[i] = party_list.index(i)
    new_member_info = {}
    for i in member_info:
        new_member_info[member_dict[i]] = [state_dict[member_info[i][0]], party_dict[member_info[i][1]]]
    # get adjacent matrix
    count1 = 0
    count2 = 0
    count3 = 0
    adjacent_matrix = np.zeros((len(member_list), len(member_list)), dtype=int)
    edge_list = []
    for legis in tqdm(legislation_list):
        if legis in legislation:
            for i in range(len(legislation[legis]['cosponsors'])):
                for g in range(len(legislation[legis]['cosponsors'])):
                    #if i == g:
                    #    continue
                    temp1 = legislation[legis]['cosponsors'][i][1][1]
                    temp2 = legislation[legis]['cosponsors'][g][1][1]
                    if temp1 in member_dic_reverse and temp2 in member_dic_reverse:
                        if member_dic_reverse[temp1] in member_list and member_dic_reverse[temp2] in member_list:
                            #if (member_dict[member_dic_reverse[temp1]],member_dict[member_dic_reverse[temp2]]) not in edge_list:
                            edge_list.append((member_dict[member_dic_reverse[temp1]],member_dict[member_dic_reverse[temp2]]))
                            count1 += 1
                            adjacent_matrix[member_dict[member_dic_reverse[temp1]]][member_dict[member_dic_reverse[temp2]]] += 1
                            adjacent_matrix[member_dict[member_dic_reverse[temp2]]][member_dict[member_dic_reverse[temp1]]] += 1
                        else:
                            count3 += 1
                    else:
                        count2 += 1

    # eliminate some edge in the adj matrix
    if eliminate:
        flatten_adj = adjacent_matrix.flatten()
        flatten_adj = np.ma.masked_equal(flatten_adj,0.0).compressed()
        median_edge = np.median(flatten_adj)
        adjacent_matrix[adjacent_matrix<=median_edge] = 0
    
    print('Data preparation finished!')
    
    # cache
    save_obj(train_vote,'train_vote') 
    save_obj(test_vote,'test_vote')
    save_obj(member_dict,'member_dict')
    save_obj(state_dict,'state_dict')
    save_obj(party_dict,'party_dict')
    save_obj(edge_list,'edge_list')
    np.save('adj_matrix',adjacent_matrix)
    if graph:
        return train_vote, test_vote, member_dict, state_dict, party_dict, adjacent_matrix, edge_list
    return train_vote, test_vote, member_dict, state_dict, party_dict, new_member_info, adjacent_matrix

'''
with open('senate_vote.json') as g:
    g = json.load(g)

# #time
# for i in g:
# 	print(g[i]['date'].split(' ')[2][:-1])

for i in g:
    for member in g[i]['member']:
        for name in member:
            print(name)
            print(member[name]['state'])
            print(member[name]['vote'])
            print(member[name]['party'])
            '''