import argparse
import json
from pprint import pprint
from time import sleep
from pathlib import Path

from blink import main_dense as blink
from blink import ner as NER
from torch.utils.data import DataLoader

from ..datasets.aida import AidaDatasetTitles
from ..datasets.createDataset import getDataset
from .metrics import calculate_recall_at
from .. import celr_module_dir


# dataset provided needs to have columns for "text" and 
# "mentions" (array with start, end, mention, url).
def generate_data_to_link_blink(ds, title2id=None):
    data_to_link = []
    # iterate over sentences
    for x in ds:
        text = x["text"]
        mentions = x["mentions"]

        # iterate over mentions in the sentence
        left_context = ""

        for m in mentions:
            start = int(m[0])
            end = int(m[1])
            url = m[3]
            title = m[4]
            d = {
                "id": len(data_to_link),
                "context_left":text[:start-1] if start > 0 else "",
                "mention":text[start:end],
                "Wikipedia_URL": url,
                "Wikipedia_title": title,
                "context_right":text[end:],
            }
            
            if title2id:
                idx = title2id[title]
                d["label_id"] = int(idx)
                d["label"] = str(idx)
            else:
                d["label_id"] = -1
                d["label"] = "unknown"
            
            data_to_link.append(d)
    return data_to_link


def eval_blink(basedir, args):
    # load datasets
    print("Load our dataset:")
    ds = getDataset(
        None,
        Path(args.dataset_cache_dir),
        Path(args.kg_ds_dir), 
        "entity-linking-untokenized-title", 
        args.num_processes, 
        args.force_exept_query, 
        args.force
    )
    print(ds)
    print(ds[0])

    print("Load Aida dataset:")
    aida_ds = AidaDatasetTitles(args, Path(args.kg_cache_dir) / "wiki/")
    print(aida_ds)
    print(aida_ds[0])

    # load model
    print("Loading model:")
    models_path = str(celr_module_dir / ".." / "BLINK/models/") + "/" # the path where you stored the BLINK models
    print(models_path+"biencoder_wiki_large.bin")

    config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": 10,
        "biencoder_model": models_path+"biencoder_wiki_large.bin",
        "biencoder_config": models_path+"biencoder_wiki_large.json",
        "entity_catalogue": models_path+"entity.jsonl",
        "entity_encoding": models_path+"all_entities_large.t7",
        "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
        "crossencoder_config": models_path+"crossencoder_wiki_large.json",
        "fast": False, # set this to be true if speed is a concern
        "output_path": str(basedir / "logs/") # logging directory
    }

    blink_args = argparse.Namespace(**config)

    models = blink.load_models(blink_args, logger=None)
    
    # evaluate
    print("\nEvaluating:")
    print("Ours:")
    eval_blink_ds(basedir, ds, models, blink_args)
    print("\nAida:")
    eval_blink_ds(basedir, aida_ds, models, blink_args)


def eval_blink_ds(basedir, ds, models, args):
    (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = models

    data_to_link = generate_data_to_link_blink(ds)
    pprint(data_to_link[0])

    (
        biencoder_accuracy,
        recall_at,
        crossencoder_normalized_accuracy,
        overall_unormalized_accuracy,
        num_datapoints,
        predictions,
        scores,
    ) = blink.run(args, None, *models, test_data=data_to_link)
    
    print("blink output:")
    print(f"biencoder accuracy= {biencoder_accuracy}")
    print(f"recall at {args.top_k}= {recall_at}")
    print(f"crossencoder_normalized_accuracy= {crossencoder_normalized_accuracy}")
    print(f"overall_unormalized_accuracy= {overall_unormalized_accuracy}")
    print(f"support= {num_datapoints}")
    # print(f"predictions= {predictions}")
    # print(f"scores= {scores}")

    labels = [d["Wikipedia_title"] for d in data_to_link]

    recall_at = calculate_recall_at(args.top_k, labels, predictions)
    accuracy = calculate_recall_at(1, labels, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Recall@{args.top_k}: {recall_at}")
    





