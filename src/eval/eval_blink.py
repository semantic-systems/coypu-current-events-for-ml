import blink.main_dense as blink
import blink.ner as NER
from src.createDataset import getDataset
from src.datasets.aida import generate_data_to_link_blink, AidaDataset
from torch.utils.data import DataLoader
import argparse
from pprint import pprint
from time import sleep
import json


def eval_blink(basedir, args):

    # load datasets
    ds = getDataset(
        basedir, 
        None, 
        basedir / args.kg_ds_dir, 
        "entity-linking-untokenized", 
        args.num_processes, 
        args.force_exept_query, 
        args.force,
        True
    )
    print("Our dataset:", ds)

    aida_ds = AidaDataset(basedir)
    print(aida_ds)

    # load model
    models_path = str(basedir / "BLINK/models/") + "/" # the path where you stored the BLINK models
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
    
    eval_blink_ds(basedir, ds, "our", models, blink_args)

    

def eval_blink_ds(basedir, ds, ds_name, models, args):
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
    
    print(ds_name, "dataset:")
    print(f"biencoder accuracy= {biencoder_accuracy}")
    print(f"recall at {args.top_k}= {recall_at}")
    print(f"crossencoder_normalized_accuracy= {crossencoder_normalized_accuracy}")
    print(f"overall_unormalized_accuracy= {overall_unormalized_accuracy}")
    print(f"support= {num_datapoints}")
    print(f"predictions= {predictions}")
    print(f"scores= {scores}")


