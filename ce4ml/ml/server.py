import os.path

from flask import abort, Flask, jsonify, request
from flask_healthz import healthz

import sys
from os.path import dirname, abspath, curdir
sys.path.append(abspath(curdir))

from ce4ml.ml.pl_model import LocationExtractor
from transformers import AutoTokenizer
import torch
from lightning import Trainer
import argparse


# params
port = 5283
model_path = "server.ckpt"

app = Flask(__name__)
app.register_blueprint(healthz, url_prefix="/healthz")

def liveness():
    pass

def readiness():
    pass

app.config.update(
    HEALTHZ = {
        "live": app.name + ".liveness",
        "ready": app.name + ".readiness"
    }
)

location_extractor = LocationExtractor.load_from_checkpoint(model_path)
location_extractor.eval()
tokenizer = AutoTokenizer.from_pretrained(location_extractor.hparams.model_name_or_path, use_fast=True)

@app.route('/text', methods=['POST'])
def text():
    if not request.json or not 'message' in request.json:
        print(request.json)
        abort(400)

    message = request.json['message']

    tokens_batch, predictions_batch = location_extractor.infer_batch(message, tokenizer)
    
    response = {
        'tokens': tokens_batch,
        'predictions': predictions_batch,
    }
    return jsonify(response), 200

@app.route('/words', methods=['POST'])
def words():
    if not request.json or not 'message' in request.json:
        print(request.json)
        abort(400)

    message = request.json['message']

    tokens_batch, predictions_batch, word_ids_batch = location_extractor.infer_words_batch(message, tokenizer)
    
    response = {
        'tokens': tokens_batch,
        'predictions': predictions_batch,
        'word_ids': word_ids_batch,
    }
    return jsonify(response), 200


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--model_checkpoint', action='store', type=str, default="server.ckpt")
    # parser.add_argument('--port', action='store', type=int, default=5300)
    # args = parser.parse_args()

    app.run(host='0.0.0.0', port=port)

