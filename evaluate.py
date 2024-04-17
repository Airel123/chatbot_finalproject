from model import *
from preprocessing import *
import torch
import os

# 读入数据
dataClass = Corpus("nocounsel.csv", maxSentenceWordsNum=100)

model_path = "testc.pkl"
# 模型验证
chatBot = ChatBot(model_path,device=torch.device('cuda:0'))
val_bleu_score, val_avgLoss = chatBot.evaluate(dataClass, batchSize=16, streamType='test')
print("val_bleu_score", val_bleu_score)
print("val_avgLoss", val_avgLoss)

print("apply postprocessing")
val_bleu_score_post, val_avgLoss_post = chatBot.evaluate_with_postprocessing(dataClass, batchSize=16, streamType='test')
print("val_bleu_score", val_bleu_score_post)
print("val_avgLoss", val_avgLoss_post)
