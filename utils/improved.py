from sklearn.ensemble import RandomForestClassifier
import logging
from collections import Counter


class Improved(object):

	def __init__(self, language):
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		# Used for counting the occurrences of all words in trainset
		self.word_dict = Counter()
		# Used for counting the occurrences of all letters in trainset
		self.w1_dict = Counter()
		# Used for counting the occurrences of all letters witn binary combination in trainset
		self.w2_dict = Counter()
		# Used for counting the occurrences of all letters witn ternary combination in trainset
		self.w3_dict = Counter()
		self.language = language
		# from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
		if language == 'english':
			self.avg_word_length = 5.3
		else:  # spanish
			self.avg_word_length = 6.2
		self.model = RandomForestClassifier()

	def stat(self, sents):
		# Execute counting
		for sent in sents:
			wdls = sent.split(' ')
			for wd in wdls:
				self.word_dict[wd] += 1
				for lte in wd:
					self.w1_dict[lte] += 1
				for i in range(len(wd) - 1):
					tp = wd[i:i+2]
					self.w2_dict[tp] += 1
				for i in range(len(wd) - 2):
					tp = wd[i:i+3]
					self.w3_dict[tp] += 1

	def extract_features(self, word):
		# Pass a complex phrase and extract feature

		ws = word.split(' ')
		# Two features from baseline system
		len_chars = len(word) / self.avg_word_length
		len_tokens = len(ws)
		# Count the occurrence of every word
		ls = list(map(lambda wd: self.word_dict[wd], ws))
		# Count the occurrence of every letter
		ls1 = list(map(lambda wd: self.w1_dict[wd], "".join(ws)))
		# Count the frequences of all letters witn binary combination
		ls2 = [ws[i][j:j + 2] for i in range(len(ws)) for j in range(len(ws[i]) - 1)]
		# Count the frequences of all letters witn ternary combination
		ls3 = [ws[i][j:j + 3] for i in range(len(ws)) for j in range(len(ws[i]) - 2)]

		ls2 = list(map(lambda wd: self.w2_dict[wd], ls2))
		ls3 = list(map(lambda wd: self.w3_dict[wd], ls3))

		# Prevent exception
		if len(ls2) == 0:
			ls2 = [0]
		if len(ls3) == 0:
			ls3 = [0]
		# Count capitalized words
		isUp = list(map(lambda wd: 1 if len(wd) > 0 and wd[0].isupper() else 0, ws))
		# Return f1 - f8
		res = [len_chars, len(word), len_tokens, sum(ls)*1.0/len(ls), sum(ls1)*1.0/len(ls1),
				sum(ls2)*1.0/len(ls2), sum(ls3)*1.0/len(ls3), sum(isUp)*1.0/len(isUp)]
		return res

	def train(self, trainset):
		self.stat(list(set(map(lambda tr: tr['sentence'], trainset))))
		X = []
		y = []
		for sent in trainset:
			x = []
			# target_word feature
			x.extend(self.extract_features(sent['target_word']))
			X.append(x)

			y.append(sent['gold_label'])
		self.model.fit(X, y)

	def test(self, testset):
		X = []
		for sent in testset:
			x = []
			x.extend(self.extract_features(sent['target_word']))
			X.append(x)

		return self.model.predict(X)