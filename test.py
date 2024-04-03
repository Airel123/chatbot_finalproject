# 导入库
from model import ChatBot
import torch

modelPath = "model.pkl"
if __name__ == "__main__":
    print('Loading the model...')
    chatBot = ChatBot(modelPath,device=torch.device('cpu'))
    print('Finished...')
    print('chatbot: hello, nice to meet you!')

    while True:
        inputSeq = input("user: ")
        if inputSeq == 'quit':
            break
        else:
            outputSeq = chatBot.predictByBeamSearch(inputSeq)
            print('bot: ', outputSeq)
        print()
