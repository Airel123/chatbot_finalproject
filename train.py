from wv_model import *
from wv_preprocessing import *
import torch
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 读入数据
dataClass = Corpus("nocounsel.csv", maxSentenceWordsNum=100)

# 指定模型和一些超参
featureSize = 128
hiddenSize = 512
encoderNumLayers = 3
decoderNumLayers = 2
dropout = 0.4588239706423144
learning_rate = 0.0029162932770102692

word2vec_modelPath="word2vec.model"

model = Seq2Seq(dataClass, word2vec_modelPath,featureSize=featureSize, hiddenSize=hiddenSize,
                learning_rate=learning_rate,
                encoderNumLayers=encoderNumLayers, decoderNumLayers=decoderNumLayers,
                dropout=dropout,
                device=torch.device('cuda:0'))

batchSize = 64
epoch = 1000
model.train(batchSize=batchSize, epoch=epoch)
model_path = f"model_batchsize{batchSize}_epoch{epoch}.pkl"
model.save(model_path)
