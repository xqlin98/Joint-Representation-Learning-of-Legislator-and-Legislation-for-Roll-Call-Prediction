from pprint import pprint
import json
import nltk
from nltk.corpus import stopwords
import numpy as np

vote_dict = {'Yea': 0, 'Aye': 0, 'Nay': 1, 'No': 1, 'Not Voting': 2, 'Present': 2}


def get_word_dict(self):
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



def get_all_data(time_limit_begin, time_limit_end):
	ccc0 = 0
	ccc1 = 0
	ccc2 = 0
	vote_len = 0
	with open('house_vote.json') as f:
		f = json.load(f)
	with open('senate_vote.json') as g:
		g = json.load(g)
	with open('legislation.json') as legislation:
		legislation = json.load(legislation)
	with open('member_dic.json') as member_dic:
		member_dic = json.load(member_dic)
	member_dic_reverse = {v[0][0]: k for k, v in member_dic.items()}
	legislation_list = []
	member_dict = {}
	state_dict = {}
	party_dict = {}
	train_vote = {}
	test_vote = {}
	for i in f:
		year = eval(f[i]['basic_information']['Date'].split('-')[-1])
		if year >= time_limit_begin and year < time_limit_end:
			legislation_list.append('.'.join(i.split(' ')))
			if '.'.join(i.split(' ')) not in legislation or len(legislation['.'.join(i.split(' '))]) == 0:
				continue
			train_vote['.'.join(i.split(' '))] = f[i]
			for member in f[i]:
				if member == 'basic_information' or f[i][member]['Vote'] not in vote_dict:
					continue
				member_dict[member] = 0
				state_dict[f[i][member]['State']] = 0
				party_dict[f[i][member]['Party']] = 0
				# if vote_dict[f[i][member]['Vote']] == 0:
				# 	ccc0 += 1
				# elif vote_dict[f[i][member]['Vote']] == 1:
				# 	ccc1 += 1
				# elif vote_dict[f[i][member]['Vote']] == 2:
				# 	ccc2 += 1
				# else:
				# 	print('fuck')
		elif year == time_limit_end:
			legislation_list.append('.'.join(i.split(' ')))
			if '.'.join(i.split(' ')) not in legislation or len(legislation['.'.join(i.split(' '))]) == 0:
				continue
			test_vote['.'.join(i.split(' '))] = f[i]
			for member in f[i]:
				if member == 'basic_information' or f[i][member]['Vote'] not in vote_dict:
					continue
				member_dict[member] = 0
				state_dict[f[i][member]['State']] = 0
				party_dict[f[i][member]['Party']] = 0
				if vote_dict[f[i][member]['Vote']] == 0:
					ccc0 += 1
				elif vote_dict[f[i][member]['Vote']] == 1:
					ccc1 += 1
				elif vote_dict[f[i][member]['Vote']] == 2:
					ccc2 += 1
				else:
					print('fuck')
				vote_len += 1

	for i in g:
		year = eval(g[i]['date'].split(' ')[2][:-1])
		if year >= time_limit_begin and year < time_limit_end:
			legislation_list.append(''.join(i.split(' ')))
			temp = {}
			for item in g[i]['member']:
				for key in item:
					temp[key] = {'Party': item[key]['party'], 'State': item[key]['state'], 'Vote': item[key]['vote']}
			if ''.join(i.split(' ')) not in legislation or len(legislation[''.join(i.split(' '))]['summary']) == 0:
				continue
			train_vote[''.join(i.split(' '))] = temp
			for name in temp:
				member_dict[name] = 0
				state_dict[temp[name]['State']] = 0
				party_dict[temp[name]['Party']] = 0
					# if member[name]['vote'] in vote_dict:
					# 	if vote_dict[member[name]['vote']] == 0:
					# 		ccc0 += 1
					# 	elif vote_dict[member[name]['vote']] == 1:
					# 		ccc1 += 1
					# 	elif vote_dict[member[name]['vote']] == 2:
					# 		ccc2 += 1
					# 	else:
					# 		print('fuck')
					# else:
					# 	print(member[name]['vote'])

		elif year == time_limit_end:
			legislation_list.append(''.join(i.split(' ')))
			temp = {}
			for item in g[i]['member']:
				for key in item:
					temp[key] = {'Party': item[key]['party'], 'State': item[key]['state'], 'Vote': item[key]['vote']}
			if ''.join(i.split(' ')) not in legislation or len(legislation[''.join(i.split(' '))]['summary']) == 0:
				continue
			test_vote[''.join(i.split(' '))] = temp
			for name in temp:
				member_dict[name] = 0
				state_dict[temp[name]['State']] = 0
				party_dict[temp[name]['Party']] = 0
				if temp[name]['Vote'] in vote_dict:
					if vote_dict[temp[name]['Vote']] == 0:
						ccc0 += 1
					elif vote_dict[temp[name]['Vote']] == 1:
						ccc1 += 1
					elif vote_dict[temp[name]['Vote']] == 2:
						ccc2 += 1
					else:
						print('fuck')
					vote_len += 1
				else:
					print(temp[name]['Vote'])

	member_list = list(member_dict)
	state_list = list(state_dict)
	party_list = list(party_dict)

	for i in member_dict:
		member_dict[i] = member_list.index(i)
	for i in state_dict:
		state_dict[i] = state_list.index(i)
	for i in party_dict:
		party_dict[i] = party_list.index(i)

	# get adjacent matrix
	count1 = 0
	count2 = 0
	count3 = 0
	adjacent_matrix = np.zeros((len(member_list), len(member_list)), dtype=int)
	for legis in legislation_list:
		if legis in legislation:
			for i in range(len(legislation[legis]['cosponsors'])):
				for g in range(len(legislation[legis]['cosponsors'])):
					temp1 = legislation[legis]['cosponsors'][i][1][0]
					temp2 = legislation[legis]['cosponsors'][g][1][0]
					if temp1 in member_dic_reverse and temp2 in member_dic_reverse:
						if member_dic_reverse[temp1] in member_list and member_dic_reverse[temp2] in member_list:
							count1 += 1
							adjacent_matrix[member_dict[member_dic_reverse[temp1]]][member_dict[member_dic_reverse[temp2]]] = 1
							adjacent_matrix[member_dict[member_dic_reverse[temp2]]][member_dict[member_dic_reverse[temp1]]] = 1
						else:
							count3 += 1
					else:
						count2 += 1
	print(adjacent_matrix)
	print(count1)
	print(count2)
	print(count3)
	print(ccc0)
	print(ccc1)
	print(ccc2)
	print(vote_len)
	return train_vote, test_vote, member_dict, state_dict, party_dict, adjacent_matrix


# with open('senate_vote.json') as g:
# 	g = json.load(g)

# #time
# for i in g:
# 	print(g[i]['date'].split(' ')[2][:-1])

# for i in g:
# 	for member in g[i]['member']:
# 		for name in member:
# 			print(name)
# 			print(member[name]['state'])
# 			print(member[name]['vote'])
# 			print(member[name]['party'])

# with open('house_vote.json') as f:
# 	f = json.load(f)
# with open('senate_vote.json') as g:
# 	g = json.load(g)
#
# count = 0
# count2 = 0
# for i in f:
# 	count += 1
# 	if count == 1:
# 		print(f[i])
# for i in g:
# 	count2 += 1
# 	if count2 == 1:
# 		print(g[i]['member'])

# with open('legislation.json') as leg:
# 	leg = json.load(leg)
#
# count = 0
# for i in leg:
# 	if len(leg[i]['summary']) == 0:
# 		count += 1
# print(count)