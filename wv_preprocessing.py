import csv
import re, nltk, random, time
from nltk.corpus import stopwords
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# 假设 tokenize 和 filter_sent 函数已经定义

class Corpus:
    def __init__(self, filePath, maxSentenceWordsNum=100, vector_size=100, window=5, min_count=1, workers=4):
        self.cleaned_patterns = []  # 用于存储第一列清理后数据
        self.cleaned_responses = []  # 用于存储第二列清理后数据
        with open(filePath, 'r', encoding='utf8', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    self.cleaned_patterns.append(filter_sent(row[0]))
                    self.cleaned_responses.append(filter_sent(row[1]))
                else:
                    print(f"Skipped a line with unexpected number of columns: {row}")

        self.tokenized_patterns = [tokenize(sentence) for sentence in self.cleaned_patterns]
        self.tokenized_responses = [tokenize(sentence) for sentence in self.cleaned_responses]

        # 添加特殊标识符
        self.tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
        for i in range(len(self.tokenized_patterns)):
            self.tokenized_patterns[i] = ['<SOS>'] + self.tokenized_patterns[i] + ['<EOS>']

        for i in range(len(self.tokenized_responses)):
            self.tokenized_responses[i] = ['<SOS>'] + self.tokenized_responses[i] + ['<EOS>']

        # 构建问答对列表
        data = []
        for p, r in zip(self.tokenized_patterns, self.tokenized_responses):
            if len(p) < maxSentenceWordsNum and len(r) < maxSentenceWordsNum:
                data.append(p)
                data.append(r)

        self.chatDataWord = data
        self.totalSampleNum = len(data)
        print("Total qa pairs num:", self.totalSampleNum)

        # Train Word2Vec model
        self.model = Word2Vec(sentences=self.chatDataWord, vector_size=vector_size, window=window, min_count=min_count,
                              workers=workers)
        print("word2vec model built!")

        self._word_id_map()

        try:
            chatDataId = [[[self.word2id[w] for w in qa[0]], [self.word2id[w] for w in qa[1]]] for qa in
                          self.chatDataWord]
        except:
            chatDataId = [[[self.word2id[w] for w in qa[0] if w in self.id2word],
                           [self.word2id[w] for w in qa[1] if w in self.id2word]] for qa in self.chatDataWord]

        self.QChatDataId, self.AChatDataId = [qa[0] for qa in chatDataId], [qa[1] for qa in chatDataId]
        self.trainIdList, self.testIdList = train_test_split([i for i in range(self.totalSampleNum)],
                                                             test_size=0.15)
        self.trainSampleNum, self.testSampleNum = len(self.trainIdList), len(self.testIdList)
        print(f"train pairs size: {self.trainSampleNum}; test pairs size: {self.testSampleNum}")
        print("Finished loading corpus!")

    # 构建批次（Batching）
    def batching(self, batchSize=128, type='train'):
        # Choose the correct ID list based on the type parameter
        idList = self.trainIdList if type == 'train' else self.testIdList

        # Initialize batches
        encoder_batches = []
        decoder_batches = []

        # Loop through the ID list and create batches
        for i in range(0, len(idList), batchSize):
            batch_ids = idList[i:i + batchSize]
            encoder_batch = [torch.tensor(self.QChatDataId[idx], dtype=torch.long) for idx in batch_ids]
            decoder_batch = [torch.tensor(self.AChatDataId[idx], dtype=torch.long) for idx in batch_ids]

            # Pad sequences in the batch
            encoder_batch_padded = pad_sequence(encoder_batch, batch_first=True, padding_value=self.word2id['<PAD>'])
            decoder_batch_padded = pad_sequence(decoder_batch, batch_first=True, padding_value=self.word2id['<PAD>'])

            encoder_batches.append(encoder_batch_padded)
            decoder_batches.append(decoder_batch_padded)

        return encoder_batches, decoder_batches

    def _word_id_map(self):
        # 预先定义特殊词汇的索引
        self.word2id = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<SOS>': 3}
        self.id2word = {0: '<PAD>', 1: '<EOS>', 2: '<UNK>', 3: '<SOS>'}

        # 从Word2Vec模型的词汇表中获取词汇，并更新映射
        # 由于已经添加了4个特殊标识，所以这里的索引需要从4开始
        next_index = 4
        for word in self.model.wv.index_to_key:
            if word not in self.word2id:
                self.word2id[word] = next_index
                self.id2word[next_index] = word
                next_index += 1
        print("Word-ID mapping built!")
        print(self.word2id)
        self.wordNum = len(self.id2word)
        print('Unique words num:', len(self.id2word) - 4)


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def filter_sent(txt):
    # lower
    txt = txt.lower()
    txt = re.sub(r'\d+', '', txt)
    txt = re.sub(r'\n', " ", txt)
    txt = re.sub(r"_", " ", txt)
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt
