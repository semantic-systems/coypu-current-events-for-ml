from ..datasets.createDataset import getDataset
from ..datasets.aida import *
from pprint import pprint
import argparse

from .metrics import print_metrics

# https://github.com/facebookresearch/BLINK/tree/main/elq
from elq import main_dense as elq


# dataset provided needs to have columns for "text" and 
# "mentions" (array with start, end, mention, url, title).
def generate_data_to_link_elq(ds):
    # build dataset
    data_to_link = []

    i = 0 # id counter 
    # iterate over sentences
    for x in ds:
        text = x["text"]
        mentions = x["mentions"]

        # iterate over mentions in the sentence
        elq_mentions = []
        titles = []
        for m in mentions:
            start = int(m[0])
            end = int(m[1])
            url = m[3]
            title = m[4]
            elq_mentions.append([start, end])
            titles.append(title)
        
        # append
        data_to_link.append({
            "id": i,
            "text": text,
            "mentions": elq_mentions,
            "entity": titles,
        })
        i += 1
    return data_to_link




def eval_elq(basedir, args):
    # Load dataset
    ds = getDataset(
        None,
        Path(args.dataset_cache_dir),
        Path(args.kg_ds_dir),
        "entity-linking-untokenized-title",
        args.num_processes,
        args.force_exept_query,
        args.force
    )
    print("Our dataset:")
    print(ds)
    print(ds[0])

    aida_ds = AidaDatasetTitles(args, Path(args.kg_cache_dir) / "wiki/")
    print(aida_ds)
    print(aida_ds[0])

    # Load Model
    models_path = str(basedir / "BLINK/models/") + "/" # the path where you stored the ELQ models

    config = {
        "interactive": False,
        "biencoder_model": models_path+"elq_wiki_large.bin",
        "biencoder_config": models_path+"elq_large_params.txt",
        "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
        "entity_catalogue": models_path+"entity.jsonl",
        "entity_encoding": models_path+"all_entities_large.t7",
        "output_path": str(basedir / "logs/"), # logging directory
        "faiss_index": "hnsw",
        "index_path": models_path+"faiss_hnsw_index.pkl",
        "num_cand_mentions": 10,
        "num_cand_entities": 10,
        "threshold_type": "joint",
        "threshold": -4.5,
    }

    elq_args = argparse.Namespace(**config)
    models = elq.load_models(elq_args, logger=None)

    # evaluate
    print("\nEvaluating:")
    print("Ours:")
    eval_elq_ds(basedir, ds, models, elq_args)
    print("\nAida:")
    eval_elq_ds(basedir, aida_ds, models, elq_args)


def eval_elq_ds(basedir, ds, models, args):
    data_to_link = generate_data_to_link_elq(ds)
    pprint(data_to_link[0])
    
    pred_cache_path = Path(args.cache_dir) / "elq_predictions.json"
    if exists(pred_cache_path):
        with open(pred_cache_path, "w", encoding="utf-8") as f:
            predictions = json.load(fp)
    else:
        predictions = elq.run(args, None, *models, test_data=data_to_link)
        with open(pred_cache_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f)

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for prediction in predictions:
        idx = prediction["id"]
        found_entities = [ x[0] for x in prediction["pred_tuples_string"] ]

        true_entities = data_to_link[i]["entity"]

        temp_entities = true_entities

        # calculate positives
        for found_entity in found_entities:
            if found_entity in true_entities:
                true_pos += 1
            else:
                false_pos += 1
            temp_entities.remove(found_entity)

        # calculate negatives
        for e in temp_entities:
            if e in true_entities:
                false_neg += 1
            else:
                true_neg += 1

    print_metrics(true_pos, false_pos, true_neg, false_neg)
