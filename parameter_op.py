from model import *
from preprocessing import *
import torch
import os
import nltk
import optuna


nltk.download('punkt')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# 定义一个日志函数，用于同时打印消息到控制台和写入到日志文件
def log(message, file_path="training_log.txt"):
    print(message)
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")


def objective(trial):
    # 定义超参数的搜索空间
    featureSize = trial.suggest_categorical('featureSize', [128, 256, 512])
    hiddenSize = trial.suggest_categorical('hiddenSize', [128, 256, 512])
    encoderNumLayers = trial.suggest_int('encoderNumLayers', 2, 5)
    decoderNumLayers = trial.suggest_int('decoderNumLayers', 2, encoderNumLayers)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    # 读入数据
    dataClass = Corpus("finaldataset0310.csv", maxSentenceWordsNum=100)

    # 初始化和训练Seq2Seq模型
    model = Seq2Seq(dataClass, featureSize=featureSize, hiddenSize=hiddenSize,
                    learning_rate=learning_rate,
                    encoderNumLayers=encoderNumLayers, decoderNumLayers=decoderNumLayers,
                    dropout=dropout,
                    device=torch.device('cuda:0'))
    model.train(batchSize=1024, epoch=10)

    model_path = "model_temple.pkl"  # 更新文件名以保持一致
    model.save(model_path)

    # 使用ChatBot类加载模型并进行评估
    chat_bot = ChatBot(modelPath=model_path, device=torch.device('cuda:0'))
    val_bleu_score, val_avgLoss = chat_bot.evaluate(dataClass, batchSize=64, streamType='test')

    # 记录结果
    log(f"Trial {trial.number}: BLEU score = {val_bleu_score}")

    # 根据BLEU分数保存模型和参数信息
    if val_bleu_score > 0.003:
        model.save(f"model_{trial.number}.pkl")
        with open(f"model_params.txt", "w") as params_file:
            params_file.write(f"Trial {trial.number} Parameters:\n")
            for key, value in trial.params.items():
                params_file.write(f"{key}: {value}\n")
            params_file.write(f"BLEU score: {val_bleu_score}\n")
        log(f"Model and parameters for trial {trial.number} saved.")

    return val_bleu_score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

# 记录最佳试验结果
log('Best trial information:')
best_trial = study.best_trial
# 打印最佳试验的编号，注意编号是从0开始的
log(f'Best trial number: {best_trial.number}')
# 打印最佳试验的BLEU分数值
log(f'Best trial BLEU score Value: {best_trial.value}')
# 打印最佳试验的参数
log('Best trial Params: ')
for key, value in best_trial.params.items():
    log(f'    {key}: {value}')

