import csv
import re, nltk, random
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


class Corpus:
    def __init__(self, filePath, maxSentenceWordsNum=100, id2word=None, word2id=None, wordNum=None, vector_size=100,
                 window=5, min_count=1, workers=4, testSize=0.15):

        self.id2word, self.word2id, self.wordNum = id2word, word2id, wordNum
        cleaned_patterns = []  # 用于存储第一列清理后数据
        cleaned_responses = []  # 用于存储第二列清理后数据
        with open(filePath, 'r', encoding='utf8', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    cleaned_patterns.append(filter_sent(row[0]))
                    cleaned_responses.append(filter_sent(row[1]))
                else:
                    print(f"Skipped a line with unexpected number of columns: {row}")

        tokenized_patterns = [tokenize(sentence) for sentence in cleaned_patterns]
        tokenized_responses = [tokenize(sentence) for sentence in cleaned_responses]

        # 添加特殊标识符
        self.tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
        for i in range(len(tokenized_patterns)):
            tokenized_patterns[i] = ['<SOS>'] + tokenized_patterns[i] + ['<EOS>']

        for i in range(len(tokenized_responses)):
            tokenized_responses[i] = ['<SOS>'] + tokenized_responses[i] + ['<EOS>']

        # 构建问答对列表
        data = []
        for p, r in zip(tokenized_patterns, tokenized_responses):
            if len(p) < maxSentenceWordsNum and len(r) < maxSentenceWordsNum:
                data.append(p)
                data.append(r)

        self.chatDataWord = data

        # Train Word2Vec model
        self.model = Word2Vec(sentences=self.chatDataWord, vector_size=vector_size, window=window, min_count=min_count,
                              workers=workers)
        self.model.save("word2vec.model")
        print("word2vec model built and saved!")


        self._word_id_map()

        try:
            chatDataId = [[[self.word2id[w] for w in qa[0]], [self.word2id[w] for w in qa[1]]] for qa in
                          self.chatDataWord]
        except KeyError:
            chatDataId = [[[self.word2id.get(w, self.word2id['<UNK>']) for w in qa[0]],
                           [self.word2id.get(w, self.word2id['<UNK>']) for w in qa[1]]] for qa in self.chatDataWord]

        self._QALens(chatDataId)
        self.maxSentLen = max(maxSentenceWordsNum, self.AMaxLen)
        self.QChatDataId, self.AChatDataId = [qa[0] for qa in chatDataId], [qa[1] for qa in chatDataId]
        self.totalSampleNum = len(data)
        print("Total qa pairs num:", self.totalSampleNum)
        self.trainIdList, self.testIdList = train_test_split([i for i in range(self.totalSampleNum)],
                                                             test_size=testSize)
        self.trainSampleNum, self.testSampleNum = len(self.trainIdList), len(self.testIdList)
        print(f"train pairs size: {self.trainSampleNum}; test pairs size: {self.testSampleNum}")
        self.testSize = testSize
        print("Finished loading corpus!")

    # Resets mappings between words and IDs.
    def reset_word_id_map(self, id2word, word2id):
        self.id2word, self.word2id = id2word, word2id
        # Update chat data IDs based on new mappings.
        chatDataId = [[[self.word2id.get(w, self.word2id['<UNK>']) for w in qa[0]],
                       [self.word2id.get(w, self.word2id['<UNK>']) for w in qa[1]]] for qa in self.chatDataWord]
        self.QChatDataId, self.AChatDataId = zip(*[(qa[0], qa[1]) for qa in chatDataId])

    # Generates a random batch of data for training or testing.
    def random_batch_data_stream(self, batchSize=128, type='train'):
        idList = self.trainIdList if type == 'train' else self.testIdList
        eosToken, unkToken = self.word2id['<EOS>'], self.word2id['<UNK>']
        while True:
            # Select a random subset of data for the batch.
            samples = random.sample(idList, min(batchSize, len(idList))) if batchSize > 0 else idList
            QMaxLen, AMaxLen = max(self.QLens[samples]), max(self.ALens[samples])
            # Prepare question and answer data with padding.
            QDataId = np.array([self.QChatDataId[i] + [eosToken] * (QMaxLen - self.QLens[i] ) for i in samples],
                               dtype='int32')
            ADataId = np.array([self.AChatDataId[i] + [eosToken] * (AMaxLen - self.ALens[i] ) for i in samples],
                               dtype='int32')
            yield QDataId, self.QLens[samples], ADataId, self.ALens[samples]

    # Provides data for a complete epoch, used in evaluating the model.
    def one_epoch_data_stream(self, batchSize=128, type='train'):
        idList = self.trainIdList if type == 'train' else self.testIdList
        eosToken = self.word2id['<EOS>']
        for i in range((len(idList) + batchSize - 1) // batchSize):
            # Slice data for current batch.
            samples = idList[i * batchSize:(i + 1) * batchSize]
            QMaxLen, AMaxLen = max(self.QLens[samples]), max(self.ALens[samples])
            # Prepare question and answer data with padding.
            QDataId = np.array([self.QChatDataId[i] + [eosToken] * (QMaxLen - self.QLens[i]  ) for i in samples],
                               dtype='int32')
            ADataId = np.array([self.AChatDataId[i] + [eosToken] * (AMaxLen - self.ALens[i] ) for i in samples],
                               dtype='int32')
            yield QDataId, self.QLens[samples], ADataId, self.ALens[samples]

    def _QALens(self, data):
        QLens, ALens = [len(qa[0]) for qa in data], [len(qa[1]) for qa in data]
        QMaxLen, AMaxLen = max(QLens), max(ALens)
        print('QMAXLEN:', QMaxLen, '  AMAXLEN:', AMaxLen)
        self.QLens, self.ALens = np.array(QLens, dtype='int32'), np.array(ALens, dtype='int32')
        self.QMaxLen, self.AMaxLen = QMaxLen, AMaxLen

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
        # print(self.word2id)
        self.wordNum = len(self.id2word)
        print('Unique words num:', len(self.id2word) - 4)


def seq2id(word2id, seqData):
    seqId = [word2id[w] for w in seqData]
    return seqId


def id2seq(id2word, seqId):
    seqData = [id2word[i] for i in seqId]
    return seqData


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
