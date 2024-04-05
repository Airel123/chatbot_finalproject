# 导入库
import torch
from torch import nn
from torch.nn import functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time
import pandas as pd
from preprocessing import *

# 定义开始符和结束符
sosToken = 1
eosToken = 0


# 定义Encoder
class EncoderRNN(nn.Module):
    # 初始化
    def __init__(self, featureSize, hiddenSize, embedding, numLayers=1, dropout=0.1,
                 bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.embedding = embedding
        # 核心API，建立双向GRU
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=(0 if numLayers == 1 else dropout),
                          bidirectional=bidirectional, batch_first=True)
        # 超参
        self.featureSize = featureSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional

    # 前向计算，训练和测试必须的部分
    def forward(self, input, lengths, hidden):
        # input: batchSize × seq_len; hidden: numLayers*d × batchSize × hiddenSize
        # 给定输入
        input = self.embedding(input)  # => batchSize × seq_len × feaSize
        lengths = lengths.to('cpu', dtype=torch.int64)
        # 加入paddle 方便计算
        packed = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        output, hn = self.gru(packed,hidden)  # output: batchSize × seq_len × hiddenSize*d; hn: numLayers*d × batchSize × hiddenSize
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # 双向GRU
        output = output[:, :, :self.hiddenSize] + output[:, :, self.hiddenSize:]
        return output, hn


# 定义LuongAttention
class LuongAttention(nn.Module):
    # 初始化
    def __init__(self, method, hiddenSize):
        super(LuongAttention, self).__init__()
        self.method = method
        if self.method == 'general':
            self.Wa = nn.Linear(hiddenSize, hiddenSize)
        elif self.method == 'concat':
            self.Wa = nn.Linear(hiddenSize * 2, hiddenSize)
            self.v = nn.Parameter(torch.FloatTensor(1, hiddenSize))  # self.v: 1 × hiddenSize

    # 给出dot计算方法
    def dot_score(self, hidden, encoderOutput):
        return torch.sum(hidden * encoderOutput, dim=2)

    # 给出general计算方法
    def general_score(self, hidden, encoderOutput):
        energy = self.Wa(encoderOutput)  # energy: batchSize × seq_len × hiddenSize
        return torch.sum(hidden * energy, dim=2)

    # 给出gconcat计算方法
    def concat_score(self, hidden, encoderOutput):
        # hidden: batchSize × 1 × hiddenSize; encoderOutput: batchSize × seq_len × hiddenSize
        energy = torch.tanh(self.Wa(torch.cat((hidden.expand(-1, encoderOutput.size(1), -1), encoderOutput),
                                              dim=2)))  # energy: batchSize × seq_len × hiddenSize
        return torch.sum(self.v * energy, dim=2)

    # 定义前向计算
    def forward(self, hidden, encoderOutput):
        # 确定使用哪种计算方式，3选1
        if self.method == 'general':
            attentionScore = self.general_score(hidden, encoderOutput)
        elif self.method == 'concat':
            attentionScore = self.concat_score(hidden, encoderOutput)
        elif self.method == 'dot':
            attentionScore = self.dot_score(hidden, encoderOutput)
        # attentionScore: batchSize × seq_len
        return F.softmax(attentionScore, dim=1).unsqueeze(1)  # => batchSize × 1 × seq_len


# 定义LuongAttentionDecoder
class LuongAttentionDecoderRNN(nn.Module):
    # 初始化
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1, attnMethod='concat'):
        super(LuongAttentionDecoderRNN, self).__init__()

        self.embedding = embedding
        # 对输入进行dropout
        self.dropout = nn.Dropout(dropout)
        # 核心api，搭建GRU
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=dropout,
                          batch_first=True)
        # 定义权重计算和联合方式
        self.attention_weight = LuongAttention(attnMethod, hiddenSize)
        self.attention_combine = nn.Linear(hiddenSize * 2, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.numLayers = numLayers

    # 定义前向计算
    def forward(self, inputStep, hidden, encoderOutput):
        # inputStep: batchSize × 1; hidden: numLayers × batchSize × hiddenSize
        # 对输入做dropout
        inputStep = self.embedding(inputStep)  # => batchSize × 1 × feaSize
        inputStep = self.dropout(inputStep)
        output, hidden = self.gru(inputStep,
                                  hidden)  # output: batchSize × 1 × hiddenSize; hidden: numLayers × batchSize × hiddenSize
        attentionWeight = self.attention_weight(output, encoderOutput)  # batchSize × 1 × seq_len
        # encoderOutput: batchSize × seq_len × hiddenSize
        context = torch.bmm(attentionWeight, encoderOutput)  # context: batchSize × 1 × hiddenSize
        attentionCombine = self.attention_combine(
            torch.cat((output, context), dim=2))  # attentionCombine: batchSize × 1 × hiddenSize
        attentionOutput = torch.tanh(attentionCombine)  # attentionOutput: batchSize × 1 × hiddenSize
        output = F.log_softmax(self.out(attentionOutput), dim=2)  # output: batchSize × 1 × outputSize
        return output, hidden, attentionWeight


class Seq2Seq:
    # 初始化

    def __init__(self, dataClass, featureSize, hiddenSize, encoderNumLayers=1, decoderNumLayers=1,
                 dropout=0.1, outputSize=None, embedding=None, learning_rate=0.001,
                 device=torch.device("cpu")):
        outputSize = outputSize if outputSize else dataClass.wordNum
        embedding = embedding if embedding else nn.Embedding(outputSize + 1, featureSize)
        # 数据读入
        self.dataClass = dataClass
        # 模型架构GRU
        self.hiddenSize = featureSize, hiddenSize
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.encoderRNN = EncoderRNN(featureSize, hiddenSize, embedding=embedding, numLayers=encoderNumLayers,
                                     dropout=dropout, bidirectional=True).to(device)
        self.decoderRNN = LuongAttentionDecoderRNN(featureSize, hiddenSize, outputSize, embedding=embedding,
                                                   numLayers=decoderNumLayers, dropout=dropout,
                                                   attnMethod='concat').to(device)
        self.embedding = embedding.to(device)
        self.device = device
        # 定义训练方法

    def train(self, batchSize, epoch=100, stopRound=10, betas=(0.9, 0.99), eps=1e-08, weight_decay=0,
              teacherForcingRatio=0.5):
        lr = self.learning_rate
        self.encoderRNN.train(), self.decoderRNN.train()
        # 给定batchSize和是否数据增广
        # Adjust batchSize based on available training samples
        batchSize = min(batchSize, self.dataClass.trainSampleNum) if batchSize > 0 else self.dataClass.trainSampleNum
        # Prepare training stream
        dataStream = self.dataClass.random_batch_data_stream(batchSize=batchSize)
        # 定义优化器，使用adam
        encoderOptimzer = torch.optim.Adam(self.encoderRNN.parameters(), lr=lr, betas=betas, eps=eps,
                                           weight_decay=weight_decay)
        decoderOptimzer = torch.optim.Adam(self.decoderRNN.parameters(), lr=lr, betas=betas, eps=eps,
                                           weight_decay=weight_decay)
        # Prepare testing stream
        if self.dataClass.testSize > 0:
            testStrem = self.dataClass.random_batch_data_stream(batchSize=batchSize, type='test')

        itersPerEpoch = self.dataClass.trainSampleNum // batchSize
        st = time.time()  # Start time for training

        # 做每个epoch循环
        for e in range(epoch):
            for i in range(itersPerEpoch):
                X, XLens, Y, YLens = next(dataStream)
                loss = self._train_step(X, XLens, Y, YLens, encoderOptimzer, decoderOptimzer, teacherForcingRatio)

                # Periodically evaluate bleu的
                if (e * itersPerEpoch + i + 1) % stopRound == 0:
                    bleu = _bleu_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, self.dataClass.maxSentLen,
                                       device=self.device)
                    print(f"After iters {e * itersPerEpoch + i + 1}: loss = {loss:.3f}; train bleu: {bleu:.3f}; ",
                          end='')

                    if self.dataClass.testSize > 0:
                        X, XLens, Y, YLens = next(testStrem)
                        bleu = _bleu_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens,
                                           self.dataClass.maxSentLen, device=self.device)
                        print(f'test bleu: {bleu:.3f}')

    # save model
    def save(self, path):
        torch.save({"encoder": self.encoderRNN, "decoder": self.decoderRNN,
                    "word2id": self.dataClass.word2id, "id2word": self.dataClass.id2word}, path)
        print(f'Model saved in "{path}".')

    # 训练中的梯度及loss计算
    def _train_step(self, X, XLens, Y, YLens, encoderOptimzer, decoderOptimzer, teacherForcingRatio):
        encoderOptimzer.zero_grad()
        decoderOptimzer.zero_grad()
        loss, nTotal = _calculate_loss(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, teacherForcingRatio,
                                       device=self.device)
        # 实现反向传播backpropagation
        (loss / nTotal).backward()
        # Update the parameters of both the encoder and decoder
        encoderOptimzer.step()
        decoderOptimzer.step()
        return loss.item() / nTotal


class ChatBot:
    def __init__(self, modelPath, device=torch.device('cpu')):  # 初始化
        # modelDict = torch.load(modelPath)
        modelDict = torch.load(modelPath, map_location=torch.device('cpu'))
        self.encoderRNN, self.decoderRNN = modelDict['encoder'].to(device), modelDict['decoder'].to(device)
        self.word2id, self.id2word = modelDict['word2id'], modelDict['id2word']
        self.hiddenSize = self.encoderRNN.hiddenSize
        self.device = device
        self.encoderRNN.eval(), self.decoderRNN.eval()

    # beamsearch的定义，inference时使用，计算量比贪婪算法大
    def predictByBeamSearch(self, inputSeq, beamWidth=100, maxAnswerLength=50, alpha=0.7, isRandomChoose=False,
                            improve=True, showInfo=False):
        # Initialize output vocabulary size and preprocess input sequence
        outputSize = len(self.id2word)
        inputSeq = filter_sent(inputSeq)
        inputSeq = [w for w in tokenize(inputSeq) if w in self.word2id.keys()]  # 分词
        # prepare tensors for the encoder
        X = seq2id(self.word2id, inputSeq)
        XLens = torch.tensor([len(X) + 1], dtype=torch.int, device=self.device)  # Include the end-of-sequence token
        X = X + [eosToken]  # Append end-of-sequence token to the input
        X = torch.tensor([X], dtype=torch.long, device=self.device)

        # Initialize hidden states for encoder
        d = int(self.encoderRNN.bidirectional) + 1  # Adjust for bidirectional GRU
        hidden = torch.zeros((d * self.encoderRNN.numLayers, 1, self.hiddenSize), dtype=torch.float32,
                             device=self.device)
        # Encode input sequence
        encoderOutput, hidden = self.encoderRNN(X, XLens, hidden)
        hidden = hidden[-d * self.decoderRNN.numLayers::2].contiguous()  # Adjust hidden state shape for the decoder

        # Initialize beams
        Y = np.ones([beamWidth, maxAnswerLength],
                    dtype='int32') * eosToken  # All beams initialized to end-of-sequence token
        # prob: beamWidth × 1
        prob = np.zeros([beamWidth, 1], dtype='float32')  # Probability scores for each beam

        # Start decoding
        decoderInput = torch.tensor([[sosToken]], dtype=torch.long, device=self.device)  # Start-of-sequence token
        # decoderOutput: 1 × 1 × outputSize; hidden: numLayers × 1 × hiddenSize
        decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden,
                                                                        encoderOutput)  # Initial decoder output
        # topv: 1 × 1 × beamWidth; topi: 1 × 1 × beamWidth
        topv, topi = decoderOutput.topk(beamWidth)  # Get top beamWidth predictions
        # decoderInput: beamWidth × 1
        decoderInput = topi.view(beamWidth, 1)  # Prepare next decoder input

        # Update beams with initial predictions
        for i in range(beamWidth):
            Y[i, 0] = decoderInput[i].item()
        prob += topv.view(beamWidth, 1).data.cpu().numpy()  # Update probabilities
        # Copy tensors for manipulation
        Y_ = Y.copy()
        prob_ = prob.copy()

        # Expand hidden states and encoder output for all beams
        # hidden: numLayers × beamWidth × hiddenSize
        hidden = hidden.expand(-1, beamWidth, -1).contiguous()
        localRestId = np.array(range(beamWidth), dtype='int32')
        # localRestId = np.array([i for i in range(beamWidth)], dtype='int32')
        encoderOutput = encoderOutput.expand(beamWidth, -1, -1)  # => beamWidth × 1 × hiddenSize

        # Iterate over the maximum answer length to build beams
        for i in range(1, maxAnswerLength):
            # Generate next word predictions for all beams
            # decoderOutput: beamWidth × 1 × outputSize; hidden: numLayers × beamWidth × hiddenSize; decoderAttentionWeight: beamWidth × 1 × XSeqLen
            decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden, encoderOutput)
            # topv: beamWidth × 1; topi: beamWidth × 1
            # Improve the selection process based on generated words if true解码器会综合之前已经生成的单词和当前生成的单词的可能性，然后做出最好的选择
            if improve:
                decoderOutput = decoderOutput.view(-1, 1)
                topv, topi = decoderOutput.topk(beamWidth, dim=0)
            else:
                topv, topi = (torch.tensor(prob[localRestId], dtype=torch.float32, device=self.device).unsqueeze(
                    2) + decoderOutput).view(-1, 1).topk(beamWidth, dim=0)
            # Update decoder input
            # decoderInput: beamWidth × 1
            decoderInput = topi % outputSize
            # Update beams and probabilities
            # 计算过程，主要算概率，算路径上的最大概率
            idFrom = topi.cpu().view(-1).numpy() // outputSize
            Y[localRestId, :i + 1] = np.hstack([Y[localRestId[idFrom], :i], decoderInput.cpu().numpy()])
            prob[localRestId] = prob[localRestId[idFrom]] + topv.data.cpu().numpy()
            hidden = hidden[:, idFrom, :]

            # Filter beams that have reached the end-of-sequence token
            restId = (decoderInput != eosToken).cpu().view(-1)
            localRestId = localRestId[restId.numpy().astype('bool')]
            decoderInput = decoderInput[restId]
            hidden = hidden[:, restId, :]
            encoderOutput = encoderOutput[restId]
            beamWidth = len(localRestId)
            if beamWidth < 1:  # 直到搜索宽度为0 Stop if all beams reached the end-of-sequence token
                break

        # Select the best beam based on the final scores
        lens = [i.index(eosToken) if eosToken in i else maxAnswerLength for i in Y.tolist()]
        ans = [' '.join(id2seq(self.id2word, i[:l])) for i, l in zip(Y, lens)]
        prob = [prob[i, 0] / np.power(lens[i], alpha) for i in range(len(ans))]

        # Choose answer based on probability
        if isRandomChoose:  # 对于回答方面做的策略，会去prob最大的那个，同时也可以给出概率
            prob = [np.exp(p) for p in prob]
            prob = [p / sum(prob) for p in prob]
            if showInfo:
                for i in range(len(ans)):
                    print((ans[i], prob[i]))
            return random_pick(ans, prob)
        else:
            ansAndProb = list(zip(ans, prob))
            ansAndProb.sort(key=lambda x: x[1], reverse=True)
            if showInfo:
                for i in ansAndProb:
                    print(i)
            return ansAndProb[0][0]

    def evaluate(self, dataClass, batchSize=128, streamType='train'):
        # Update word-id mappings in the dataClass with the current model's vocabulary
        dataClass.reset_word_id_map(self.id2word, self.word2id)
        # Retrieve a data stream
        dataStream = dataClass.one_epoch_data_stream(batchSize=batchSize, type=streamType)
        bleuScore, totalLoss = 0.0, 0.0
        totalSamplesNum = dataClass.trainSampleNum if streamType == 'train' else dataClass.testSampleNum
        iters, nTotal = 0, 0
        while True:
            try:
                X, XLens, Y, YLens = next(dataStream)
            except StopIteration:  # no more batches, exit the loop
                break

            # Calculate BLEU score / loss  for current batch
            bleuScore += _bleu_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, dataClass.maxSentLen,
                                     self.device)
            batchLoss, batchNTotal = _calculate_loss(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens,
                                                     teacherForcingRatio=0, device=self.device)
            totalLoss += batchLoss.item()
            nTotal += batchNTotal
            # Update the count of processed samples
            iters += len(X)

        avgBleuScore = bleuScore / totalSamplesNum
        avgLoss = totalLoss / nTotal if nTotal > 0 else 0
        print(f'Evaluation completed: Avg BLEU Score = {avgBleuScore:.4f}, Avg Loss = {avgLoss:.4f}')
        return avgBleuScore, avgLoss


def random_pick(sample, prob):  # 随机pick一个prob比较大的，根据给定的概率从一系列样本中进行加权随机选择
    x = random.uniform(0, 1)
    cntProb = 0.0
    for sampleItem, probItem in zip(sample, prob):
        cntProb += probItem
        if x < cntProb:
            break
    return sampleItem


def _bleu_score(encoderRNN, decoderRNN, X, XLens, Y, YLens, maxSentLen, device):
    Y_pre = _calculate_Y_pre(encoderRNN, decoderRNN, X, XLens, Y, maxSentLen, teacherForcingRatio=0, device=device)
    Y = [list(Y[i])[:YLens[i] - 1] for i in range(len(YLens))]
    Y_pre = Y_pre.cpu().data.numpy()
    Y_preLens = [list(i).index(0) if 0 in i else len(i) for i in Y_pre]
    Y_pre = [list(Y_pre[i])[:Y_preLens[i]] for i in range(len(Y_preLens))]

    # 使用平滑函数
    smoothie = SmoothingFunction().method1
    # `weights` defines equal importance (0.25 each) to the 1-gram, 2-gram, 3-gram, and 4-gram matchings,
    # which means it considers the match of single words up to sequences of 4 words in scoring the translation.
    bleuScore = [sentence_bleu([i], j, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie) for i, j in
                 zip(Y, Y_pre)]
    return np.mean(bleuScore)


# 计算loss
def _calculate_loss(encoderRNN, decoderRNN, X, XLens, Y, YLens, teacherForcingRatio, device):
    featureSize, hiddenSize = encoderRNN.featureSize, encoderRNN.hiddenSize
    # X: batchSize × XSeqLen; Y: batchSize × YSeqLen
    X, Y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(Y, dtype=torch.long, device=device)  # 转tensor
    XLens, YLens = torch.tensor(XLens, dtype=torch.int, device=device), torch.tensor(YLens, dtype=torch.int,
                                                                                     device=device)
    batchSize = X.size(0)
    XSeqLen, YSeqLen = X.size(1), YLens.max().item()
    encoderOutput = torch.zeros((batchSize, XSeqLen, featureSize), dtype=torch.float32, device=device)

    # Initialize hidden state （ d ->bidirectional )
    d = 2
    hidden = torch.zeros((d * encoderRNN.numLayers, batchSize, hiddenSize), dtype=torch.float32, device=device)

    # Sort X in descending order of lengths for pack_padded_sequence
    XLens, indices = torch.sort(XLens, descending=True)
    _, desortedIndices = torch.sort(indices, descending=False)
    # Encoder forward pass
    encoderOutput, hidden = encoderRNN(X[indices], XLens, hidden)
    encoderOutput, hidden = encoderOutput[desortedIndices], hidden[-d * decoderRNN.numLayers::2, desortedIndices, :]
    # Prepare the decoder input starting with sosToken for each sequence in the batch
    decoderInput = torch.tensor([[sosToken] for i in range(batchSize)], dtype=torch.long, device=device)
    loss, nTotal = 0, 0

    # Sort X in descending order of lengths for pack_padded_sequence
    # XLens, indices = torch.sort(XLens, descending=True)
    # X, Y = X[indices], Y[indices]  # Reorder X and Y according to XLens
    #
    # # Encoder forward pass
    # encoderOutput, hidden = encoderRNN(X, XLens, hidden)
    # # Prepare the decoder input starting with sosToken for each sequence in the batch
    # decoderInput = torch.tensor([[sosToken]] * batchSize, dtype=torch.long, device=device)
    # loss, nTotal = 0, 0  # Initialize loss and total count

    for i in range(YSeqLen):  # 遍历  对于每个decoder的中，都会取top，并计算loss，训练过程中对比训练数据和真实数据之间的差
        # decoderOutput: batchSize × 1 × outputSize
        decoderOutput, hidden, decoderAttentionWeight = decoderRNN(decoderInput, hidden, encoderOutput)
        # Negative Log Likelihood Loss, NLL Loss
        loss += F.nll_loss(decoderOutput[:, 0, :], Y[:, i], reduction='sum')
        nTotal += len(decoderInput)

        # Decide the next input for the decoder based on teacher forcing
        # 使用真实标签来指导模型学习
        if random.random() < teacherForcingRatio:
            decoderInput = Y[:, i:i + 1]  # Next input is current target
        else:
            # use the decoder's own prediction as the next input
            topv, topi = decoderOutput.topk(1)
            decoderInput = topi[:, :, 0]  # topi.squeeze().detach()

        restId = (YLens > i + 1).view(-1)
        decoderInput = decoderInput[restId]
        hidden = hidden[:, restId, :]
        encoderOutput = encoderOutput[restId]
        Y = Y[restId]
        YLens = YLens[restId]

    return loss, nTotal


# 计算Y的预测值
def _calculate_Y_pre(encoderRNN, decoderRNN, X, XLens, Y, YMaxLen, teacherForcingRatio, device):
    featureSize, hiddenSize = encoderRNN.featureSize, encoderRNN.hiddenSize

    # X: batchSize × XSeqLen; Y: batchSize × YSeqLen
    X, Y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(Y, dtype=torch.long, device=device)  # 给定输入
    XLens = torch.tensor(XLens, dtype=torch.int, device=device)

    batchSize = X.size(0)
    XSeqLen = X.size(1)
    encoderOutput = torch.zeros((batchSize, XSeqLen, featureSize), dtype=torch.float32, device=device)  # encoder输出
    d = 2
    hidden = torch.zeros((d * encoderRNN.numLayers, batchSize, hiddenSize), dtype=torch.float32, device=device)
    # Sort sequences
    XLens, indices = torch.sort(XLens, descending=True)
    _, desortedIndices = torch.sort(indices, descending=False)  # 排序
    # Encode input sequences
    encoderOutput, hidden = encoderRNN(X[indices], XLens, hidden)
    encoderOutput, hidden = encoderOutput[desortedIndices], hidden[-d * decoderRNN.numLayers::2, desortedIndices, :]
    # hidden[:decoderRNN.numLayers, desortedIndices, :]
    # Prepare for decoding
    decoderInput = torch.tensor([[sosToken] for i in range(batchSize)], dtype=torch.long,
                                device=device)  # 把encoder的输出接入到decoder输入中
    Y_pre = torch.ones([batchSize, YMaxLen], dtype=torch.long, device=device) * eosToken
    localRestId = torch.tensor([i for i in range(batchSize)], dtype=torch.long, device=device)
    # Decode sequence
    for i in range(YMaxLen):  # 循环 把每一个batch中的y_pre的得到（使用attention的权重）
        # decoderOutput: batchSize × 1 × outputSize
        decoderOutput, hidden, decoderAttentionWeight = decoderRNN(decoderInput, hidden, encoderOutput)
        # Apply teacher forcing
        if random.random() < teacherForcingRatio:
            decoderInput = Y[:, i:i + 1]
        else:
            topv, topi = decoderOutput.topk(1)  # 取top1
            decoderInput = topi[:, :, 0]  # topi.squeeze().detach()

        # Update predicted sequence
        Y_pre[localRestId, i] = decoderInput.squeeze()
        # Filter out sequences that have reached the end
        restId = (decoderInput != eosToken).view(-1)
        localRestId = localRestId[restId]
        decoderInput = decoderInput[restId]
        hidden = hidden[:, restId, :]
        encoderOutput = encoderOutput[restId]
        Y = Y[restId]
        # Exit loop if all sequences have ended
        if len(localRestId) < 1:
            break
    return Y_pre
