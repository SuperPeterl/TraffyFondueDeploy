# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify,render_template
import numpy as np
import onnxruntime
import json
from pythainlp.tokenize import word_tokenize
import re
import os

app = Flask(__name__)

origins = ["*"]

app.config['CORS_HEADERS'] = 'Content-Type'

model_dir = 'model'
model = onnxruntime.InferenceSession(os.path.join(model_dir, "Model.onnx"))

init_token_id = 1  #เริ่มต้นประโยค
pad_token_id = 0  #pad_id ตอนเทรนจะเติมเข้าไปกรณีบางประโยคสั้นกว่าประโยคอื่น
unk_token_id= 1 # กรณีไม่พบคำศัพท์จะแทนด้วย 2 นี้

def read_json(fname, encoding='utf-8'):
    with open(fname, encoding=encoding) as f:
        data = json.load(f)
        return data

token_to_id = read_json(os.path.join(model_dir, 'token2idx.json'))
ids_to_labs = read_json(os.path.join(model_dir, 'idx2lab.json'))

def tokens_to_ids(tokens):
    if len(tokens) == 0:
        return [0]
    
    out_id = [init_token_id]
    for w in tokens:
        if w in token_to_id.keys():
            out_id.append(token_to_id[w])
        else:
            out_id.append(unk_token_id)  # unknown word
    
    return out_id

def thai_clean_text(text):
    st = ""
    # Add more text cleaning code here, such as removing emojis
    text = text.replace("\n", " ")
    for w in word_tokenize(text):
        st = st + w + " "

    return re.sub(' +', ' ', st).strip()

@app.route('/')
def home():
    return render_template('./index.html')


@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json
    clean_text = thai_clean_text(input_data["text"])
    token_ids = tokens_to_ids(clean_text.split(' '))
    input_data = np.array([token_ids])

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: input_data.astype(np.int64)})

    lab_index = np.argmax(result[0], axis=1)
    label = ids_to_labs[str(lab_index[0])]
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
