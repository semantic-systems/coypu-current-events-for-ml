import os.path

from flask import abort, Flask, jsonify, request
from flask_healthz import healthz

import sys
from os.path import dirname, abspath, curdir
sys.path.append(abspath(curdir))

from ce4ml.ml.pl_model import LocationExtractor
from transformers import AutoTokenizer
import torch


# params
port = 5300
model_path = "ce4ml/ml/pl_trainer/lightning_logs/version_1/checkpoints/epoch=0-step=1.ckpt"

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

@app.route('/', methods=['POST'])
def flask():
    if not request.json or not 'message' in request.json:
        print(request.json)
        abort(400)

    message = request.json['message']

    tokens, predictions = location_extractor.infer([message], tokenizer)

    locations = []
    for tokens, predictions in zip(tokens_batch, predictions_batch):
        locations.append([ t for t,p in zip(tokens, predictions) if t == 1 ])
    
    response = {
        'locations': locations[0],
    }
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)

