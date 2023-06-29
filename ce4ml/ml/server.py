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
api_key = '3G65JRDTXW8QV3GJ'

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


def auth(request) -> bool:
    if 'key' in request.json and request.json['key'] == api_key:
        return True


@app.route('/', methods=['POST'])
def words_or_tokens():
    if not auth(request):
        response = {'error': 'no valid API key'}
        return jsonify(response), 401
    
    if not request.json:
        response = {'error': 'no valid input'}
        return jsonify(response), 400

    if "presplit" in message:
        presplit_inputs = request.json['presplit']
        tokens_batch, predictions_batch, word_ids_batch = location_extractor.infer_words_batch(presplit_inputs, tokenizer)
        response = {
            'tokens': tokens_batch,
            'predictions': predictions_batch,
            'word_ids': word_ids_batch,
        }
        
    elif "text" in message:
        text_inputs = request.json['text']
        tokens_batch, predictions_batch = location_extractor.infer_batch(text_inputs, tokenizer)
        response = {
            'tokens': tokens_batch,
            'predictions': predictions_batch,
        }
    
    else:
        response = {'error': 'no valid input'}
        return jsonify(response), 400
    
    return jsonify(response), 200


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--model_checkpoint', action='store', type=str, default="server.ckpt")
    # parser.add_argument('--port', action='store', type=int, default=5300)
    # args = parser.parse_args()

    app.run(host='0.0.0.0', port=port)

