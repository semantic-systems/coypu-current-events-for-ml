# ML Experiments using WikiEvents

This repo holds machine learning related experiments using WikiEvents. 

## Setup
```
./scripts/install_anaconda.sh # optional 
./scripts/install_ml.sh
./scripts/install_eval.sh
```

## Location Extractor Fine-Tuning
The `ml/` directory holds experiments for fine-tuning BERT models on WikiEvents mainly on event-related location extraction. 

To run the fine-tuning:
```
python -m ce4ml.ml
```

## Entity Linking Evaluation
The `eval/` directory holds experiments for for evaluting the capabilities of WikiEvents on providing entity linking training samples. Evalutaion is done by comparing AIDA-YAGO2 to WikiEvents using BLINK and ELQ to get performances on both. 

To run the evaluation using both models:
```
python -m ce4ml.eval -m blink
python -m ce4ml.eval -m elq
```
