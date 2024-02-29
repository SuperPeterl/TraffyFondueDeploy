# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify,render_template
import numpy as np
import onnxruntime
import json
from pythainlp.tokenize import word_tokenize
import re
import os

app = Flask(__name__,static_folder='static')

origins = ["*"]

app.config['CORS_HEADERS'] = 'Content-Type'

model_dir = 'model'
model = onnxruntime.InferenceSession(os.path.join(model_dir, "Model.onnx"))

def read_json(fname, encoding='utf-8'):
    with open(fname, encoding=encoding) as f:
        data = json.load(f)
        return data

token_to_id = read_json(os.path.join(model_dir, 'token2idx.json'))
ids_to_labs = read_json(os.path.join(model_dir, 'idx2lab.json'))
#print(token_to_id)
def process_text(text):

    text = word_tokenize(text, engine="newmm")
    print(text,end= " = ")
    text = [token_to_id.get(i, 1) for i in text]
    print(text)
    return text

@app.route('/')
def home():
    wd = "wd.jpg"
    return render_template('./index.html',wd = wd)


@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json
    text = input_data["text"]
    text = process_text(text)
    input_data = np.array([text])

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: input_data.astype(np.int64)})

    lab_index = np.argmax(result[0], axis=1)
    label = ids_to_labs[str(lab_index[0])]
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
