from flask import Flask, render_template, request, jsonify
from model import ChatBot
import torch, warnings, argparse

warnings.filterwarnings("ignore")


def get_response(inputSeq):
    # print("enter get_response function")
    modelPath = "testc.pkl"
    chatBot = ChatBot(modelPath,device=torch.device('cpu'))

    print("loading the model...")
    outputSeq = chatBot.predictByBeamSearch(inputSeq)
    return outputSeq


app = Flask(__name__)


@app.get("/")
def index_get():
    return render_template("chat_window.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    #     error checking if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
