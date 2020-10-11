import networkx as nx
import re
import numpy as np
from nltk.corpus import stopwords
import scipy.sparse
import json
from tqdm import tqdm

stws = stopwords.words('english')


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
				graph_matrix[word_list_unique.index(word_list[i])][word_list_unique.index(word_list[i - j - 1])] = 1
				graph_matrix[word_list_unique.index(word_list[i - j - 1])][word_list_unique.index(word_list[i])] = 1
			if i + j + 1 <= len(word_list) - 1:
				graph_matrix[word_list_unique.index(word_list[i])][word_list_unique.index(word_list[i + j + 1])] = 1
				graph_matrix[word_list_unique.index(word_list[i + j + 1])][word_list_unique.index(word_list[i])] = 1

	graph_matrix = scipy.sparse.csr_matrix(graph_matrix)

	nx_graph = nx.from_scipy_sparse_matrix(graph_matrix)
	scores = nx.pagerank(nx_graph)
	return [word[1][0] for word in
	        sorted([[i, [s, scores[i]]] for i, s in enumerate(word_list_unique)], key=lambda x: -x[1][1])]


def adjacentMatrix(legislation_dict):
	embeddings_index = {}
	with open('glove.6B.100d.txt', encoding='utf8') as f:
		for line in tqdm(f):
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	with open('legislation.json') as k:
		k = json.load(k)
	adj_mtx = np.zeros((len(legislation_dict), len(legislation_dict)), dtype=int)
	legislation_list = list(legislation_dict.keys())
	legislation_index_dict = {k: legislation_list.index(k) for k in legislation_list}
	legislation_index_dict_reverse = {v: k for k, v in legislation_index_dict.values()}
	legislation_rep = np.zeros(len(legislation_list), 100)
	for i in range(len(legislation_list)):
		if 'Official Title as Introduced' not in k[legislation_list[i]]['title']:
			title_key = 'Official Titles as Introduced'
		else:
			title_key = 'Official Title as Introduced'
		legislation_text = legislation_dict[legislation_list[i]]['basic_information']['Descrption'] + \
		                   k[legislation_list[i]]['title']
		keyword_list = textRank(legislation_text)
		if len(keyword_list) > 10:
			keyword_list = keyword_list[:10]
		text_vector = np.zeros(200)
		count = 0
		for kw in keyword_list:
			if kw in embeddings_index:
				text_vector += embeddings_index[kw]
				count += 1
		text_vector = text_vector / count
		legislation_index_dict_reverse[i] = text_vector
		legislation_rep[i] = text_vector
	for i in range(len(legislation_dict)):
		for j in range(len(legislation_dict)):
			adj_mtx[i][j] = cosSimilarity(legislation_index_dict_reverse[i], legislation_index_dict_reverse[j])
	return adj_mtx, legislation_index_dict, legislation_index_dict_reverse, legislation_rep


def cosSimilarity(vector1, vector2):
	cs = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
	return cs
