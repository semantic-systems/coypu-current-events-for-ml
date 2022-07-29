def eval_blink(basedir, args):
    import BLINK.blink.main_dense as blink
    import blink.ner as NER

    ds = getDataset(
        basedir, 
        None, 
        basedir / args.kg_ds_dir, 
        "entity-linking", 
        args.num_processes, 
        args.force_exept_query, 
        args.force,
        True
    )
    print("Our dataset:", ds)

    dl = DataLoader(ds)

    # TODO import aida

    """
    models_path = str(basedir / "BLINK/models/") # the path where you stored the BLINK models

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
        "output_path": basedir + "logs/" # logging directory
    }

    args = argparse.Namespace(**config)

    models = blink.load_models(args, logger=None)
    """
    for x in dl:
        data_to_link = blink._annotate(NER.model(), [x["text"]]) # extra lower()?

        _, _, _, _, _, predictions, scores, = blink.run(args, None, *models, test_data=data_to_link)





def eval_elq(ds):
    # https://github.com/facebookresearch/BLINK/tree/main/elq
    import elq.main_dense as elq

    # Load datasets
    ds = getDataset(
        basedir, 
        None, 
        basedir / args.kg_ds_dir, 
        "entity-linking", 
        args.num_processes, 
        args.force_exept_query, 
        args.force,
        True
    )
    print("Our dataset:", ds)

    dl = DataLoader(ds)

    # TODO import aida

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

    args = argparse.Namespace(**config)

    models = elq.load_models(args, logger=None)


    # predict
    data_to_link = []
    for i, row in enumerate(dl):
        data_to_link.append({ 
            "id": i,
            "text": row["text"].lower(),
        })

    predictions = main_dense.run(args, None, *models, test_data=data_to_link)

    # evaluate
    #TODO


def eval_deeptype(ds):
    pass

def eval_bert_entity(ds):
    pass
