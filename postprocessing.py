import os
from openai import OpenAI
from model import ChatBot
import torch
import datetime

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
MODEL = "gpt-3.5-turbo"


def log_dialogue(user_input, bot_output, corrected_text):
    # 定义日志文件的名称
    log_file = "dialogues.log"
    # 获取当前时间
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 准备要记录的信息，包括是否被标记为'0'
    status = "Unrealistic" if corrected_text.strip() == '0' else "Adjusted"
    log_entry = f"{now} - Status: {status} - User Input: {user_input} - Bot Output: {bot_output} - Corrected: {corrected_text}\n"

    # 将信息写入日志文件
    with open(log_file, "a") as file:
        file.write(log_entry)


def get_response(inputSeq):
    if inputSeq == 'null' or inputSeq == 'invalid':
        error_message = "Hmm, I didn't quite catch that. Could you rephrase or give me a bit more detail? I'm here to help! 🌟";
        return error_message

    modelPath = "testc.pkl"
    chatBot = ChatBot(modelPath, device=torch.device('cpu'))

    print("loading the model...")
    outputSeq = chatBot.predictByBeamSearch(inputSeq, isRandomChoose=False)
    print("outputSeq", outputSeq)
    # 使用OpenAI API纠正outputSeq的语法
    corrected_outputSeq = postprocessing(inputSeq, outputSeq)
    print("Corrected output: ", corrected_outputSeq)
    return corrected_outputSeq


def postprocessing(user_input, bot_output):
    try:
        instructions = (
            "Imagine you are an assistant whose role is to support a chatbot designed for student interactions, "
            "with a focus on mental health and stress among other topics. Your task is to evaluate the conversation "
            "based on the user input"
            "and the chatbot's predicted output. You should judge the conversation's relevance and realism. "
            "If the dialogue is something that can happen in reality and makes sense within the context of a "
            "student's life,"
            "including but not limited to mental health and stress, correct any grammatical errors in the chatbot's "
            "response,"
            "adjust casing where appropriate, add necessary punctuation, and ensure the response is accurate and "
            "sounds natural."
            "If the conversation does not make sense or could not realistically occur, return '0'."
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": instructions
                },
                {
                    "role": "user",
                    "content": user_input,
                },
                {
                    "role": "assistant",
                    "content": bot_output,
                }
            ],
            temperature=0.5,
        )

        corrected_text = response.choices[0].message.content

        if corrected_text.strip() == '0':
            print("This dialogue can't happen in reality.")
            # 如果被标记为'0'，依然记录对话，但状态为"Unrealistic"
            log_dialogue(user_input, bot_output, corrected_text)
            # 可以选择返回原始bot输出
            return bot_output
        else:
            print("User input is related, and the bot's response has been adjusted.")
            # 记录对话，状态为"Adjusted"
            log_dialogue(user_input, bot_output, corrected_text)
            return corrected_text
    except Exception as e:
        print(f"An error occurred: {e}")
        # 发生错误时，也记录这次对话尝试
        log_dialogue(user_input, bot_output, "Error occurred")
        # 在发生错误时返回原始文本
        return bot_output

# test
# get_response("hi,there")