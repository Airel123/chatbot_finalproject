import csv
import re, nltk, random
import numpy as np
from sklearn.model_selection import train_test_split  # Split dataset into training and test sets
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


# Define a class to process and prepare dialogue data for machine learning models.
class Corpus:
    # Initialize corpus with file path and optional parameters for preprocessing.
    def __init__(self, filePath, maxSentenceWordsNum=90, id2word=None, word2id=None, wordNum=None, testSize=0.15):
        self.id2word, self.word2id, self.wordNum = id2word, word2id, wordNum

        cleaned_patterns, cleaned_responses = [], []  # Lists for cleaned questions and answers

        # Read and clean dialogue data from a CSV file.
        with open(filePath, 'r', encoding='utf8', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    cleaned_patterns.append(filter_sent(row[0]))
                    cleaned_responses.append(filter_sent(row[1]))
                else:
                    print(f"Skipped a line with unexpected number of columns: {row}")

        # Tokenize and pair up questions and answers if they are under the word limit.
        tokenized_patterns = [tokenize(sentence) for sentence in cleaned_patterns]
        tokenized_responses = [tokenize(sentence) for sentence in cleaned_responses]
        data = [(p, r) for p, r in zip(tokenized_patterns, tokenized_responses) if
                len(p) < maxSentenceWordsNum and len(r) < maxSentenceWordsNum]

        self.chatDataWord = data
        self._word_id_map(data)  # Map words to IDs and vice versa

        # Convert words in questions and answers to their corresponding IDs.
        try:
            chatDataId = [[[self.word2id[w] for w in qa[0]], [self.word2id[w] for w in qa[1]]] for qa in
                          self.chatDataWord]
        except KeyError:
            chatDataId = [[[self.word2id.get(w, self.word2id['<UNK>']) for w in qa[0]],
                           [self.word2id.get(w, self.word2id['<UNK>']) for w in qa[1]]] for qa in self.chatDataWord]

        self._QALens(chatDataId)  # Calculate lengths of questions and answers
        self.maxSentLen = max(maxSentenceWordsNum, self.AMaxLen)
        self.QChatDataId, self.AChatDataId = zip(*[(qa[0], qa[1]) for qa in chatDataId])
        self.totalSampleNum = len(data)
        print("Total qa pairs num:", self.totalSampleNum)
        # Split data into training and test sets.
        self.trainIdList, self.testIdList = train_test_split(range(self.totalSampleNum), test_size=testSize)
        self.trainSampleNum, self.testSampleNum = len(self.trainIdList), len(self.testIdList)
        print("train pairs size: %d; test pairs size: %d" % (self.trainSampleNum, self.testSampleNum))
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
            QDataId = np.array([self.QChatDataId[i] + [eosToken] * (QMaxLen - self.QLens[i] + 1) for i in samples],
                               dtype='int32')
            ADataId = np.array([self.AChatDataId[i] + [eosToken] * (AMaxLen - self.ALens[i] + 1) for i in samples],
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
            QDataId = np.array([self.QChatDataId[i] + [eosToken] * (QMaxLen - self.QLens[i] + 1) for i in samples],
                               dtype='int32')
            ADataId = np.array([self.AChatDataId[i] + [eosToken] * (AMaxLen - self.ALens[i] + 1) for i in samples],
                               dtype='int32')
            yield QDataId, self.QLens[samples], ADataId, self.ALens[samples]

    # Updates lengths of questions and answers, calculating max lengths.
    def _QALens(self, data):
        QLens, ALens = [len(qa[0]) + 1 for qa in data], [len(qa[1]) + 1 for qa in data]
        QMaxLen, AMaxLen = max(QLens), max(ALens)
        print('QMAXLEN:', QMaxLen, '  AMAXLEN:', AMaxLen)
        self.QLens, self.ALens = np.array(QLens, dtype='int32'), np.array(ALens, dtype='int32')
        self.QMaxLen, self.AMaxLen = QMaxLen, AMaxLen

    # Creates or updates word-ID mappings based on the current dataset.
    def _word_id_map(self, data):
        self.id2word = list(set([w for qa in data for sent in qa for w in sent]))
        self.id2word.sort()
        self.id2word = ['<EOS>', '<SOS>'] + self.id2word + ['<UNK>']

        self.word2id = {i[1]: i[0] for i in enumerate(self.id2word)}

        self.wordNum = len(self.id2word)
        print('Unique words num:', len(self.id2word) - 3)


def seq2id(word2id, seqData):
    seqId = [word2id[w] for w in seqData]
    return seqId


def id2seq(id2word, seqId):
    seqData = [id2word[i] for i in seqId]
    return seqData


# Filters and cleans input text for preprocessing.
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


# Tokenizes sentences into words.
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
