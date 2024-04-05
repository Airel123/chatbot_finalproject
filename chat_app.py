from flask import Flask, render_template, request, jsonify
import warnings
# from postprocessing_grammarbot import get_response
# from gpt import get_response
from gpt_imrove import get_response
warnings.filterwarnings("ignore")


app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("chat_window.html")


@app.post("/predict")
# def predict():
#
#     text = request.get_json().get("message")
#     #     error checking if text is valid
#     response = get_response(text)
#     message = {"answer": response}
#     return jsonify(message)
def predict():
    json_data = request.get_json()

    # 检查是否接收到了数据
    if json_data is None:
        return jsonify({"error": "没有接收到数据"}), 400

    # 检查"message"键是否存在
    text = json_data.get("message")
    if text is None:
        return jsonify({"error": "缺少'message'字段"}), 400

    # 检查文本是否为空字符串
    if text.strip() == "":
        text = 'null'  # 特定预设文本表示输入为空

    # 检查文本是否为字符串类型
    if not isinstance(text, str):
        text = 'invalid'  # 特定预设文本表示输入类型无效

    # 如果一切正常，处理文本并返回响应
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(port=5000,debug=True)
