from src.createDataset import getDataset
from src.datasets.aida import generate_data_to_link_elq, AidaDataset
from pprint import pprint
import argparse

# https://github.com/facebookresearch/BLINK/tree/main/elq
import elq.main_dense as elq


def eval_elq(basedir, args):

    # Load datasets
    ds = getDataset(
        basedir, 
        None, 
        basedir / args.kg_ds_dir, 
        "entity-linking-untokenized-wd", 
        args.num_processes, 
        args.force_exept_query, 
        args.force,
        raw=True
    )
    print("Our dataset:", ds)
    print(ds[0])

    aida_ds = AidaDataset(basedir)
    print(aida_ds)

    # Load Model
    models_path = str(basedir / "BLINK/models/") # the path where you stored the ELQ models

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
    #models = elq.load_models(elq_args, logger=None)
    models = None

    # evaluate
    eval_elq_ds(basedir, ds, "our", models, elq_args)


def eval_elq_ds(basedir, ds, ds_name, models, args):
    data_to_link = generate_data_to_link_elq(ds)
    pprint(data_to_link[0])
   
    return # TODO delete


    predictions = elq.run(args, None, *models, test_data=data_to_link)

    #TODO look at output of elq and how to compute metrics
    
    print(ds_name, "dataset:")
    print(f"biencoder accuracy= {biencoder_accuracy}")
    print(f"recall at {config['top_k']}= {recall_at}")
    print(f"crossencoder_normalized_accuracy= {crossencoder_normalized_accuracy}")
    print(f"overall_unormalized_accuracy= {overall_unormalized_accuracy}")
    print(f"support= {num_datapoints}")

    with open(basedir / f"{ds_name}.json", "w", encoding="utf-8") as f:
        d = {
            "biencoder_accuracy":biencoder_accuracy,
            "recall_at":recall_at,
            "crossencoder_normalized_accuracy":crossencoder_normalized_accuracy,
            "overall_unormalized_accuracy":overall_unormalized_accuracy,
            "num_datapoints":num_datapoints,
            "predictions":predictions,
            "scores":scores,
        }
        json.dump(d, f, separators=(',', ':'))