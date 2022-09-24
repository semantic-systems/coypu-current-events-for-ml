import json
from glob import glob
from os import makedirs
from os.path import exists
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Union, Tuple

from datasets import ClassLabel
from datasets import Dataset as hfDataset
from datasets import DatasetDict, Features, Sequence, Value, load_dataset
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from .entity_linking import get_loc2entity
from .graph2json import (graph2json_mp_host, queryGraphEntitys,
                         queryGraphLocations)
from .queryPostprocessor import *
from .CurrentEventsDatasets import *


def tokenize_and_align_labels(examples, tokenizer, label_key):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)

    all_labels = examples[label_key]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        current_word = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                # Special token
                label_ids.append(-100)
            elif word_id != current_word:
                # Start of a new word!
                label_ids.append(labels[word_id])
            else:
                # Same word as previous token
                label_ids.append(-100)
            current_word = word_id
        new_labels.append(label_ids)

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def createJsonDataset(ds_type:str, ds_dir:Path, queryFunction, queryPostprocessor, qp_kwargs, num_processes,
        forceExeptQuery, force) -> Path:

    out_path = ds_dir / f"{ds_type}.json"
    ds_filepaths = glob(str(ds_dir / "*_*_base.jsonld"))

    if not exists(out_path) or forceExeptQuery or force:
        # import json ds
        dataset_file_paths = graph2json_mp_host(
            ds_dir,
            ds_filepaths,
            queryPostprocessor,
            queryFunction,
            num_processes,
            forceExeptQuery=forceExeptQuery,
            force=force,
            qp_kwargs=qp_kwargs
        )

        ds = load_dataset('json', data_files=dataset_file_paths, field="data")

        print(f"Save dataset to {out_path}")
        ds["train"].to_json(out_path)
    else:
        print(f"Dataset already exists at {out_path}")

    return out_path

def getDataset(basedir, tokenizer, ds_dir:Path, ds_type:str, num_processes:int,
        forceExeptQuery=False, force=False) -> Dataset:

    ds_filepaths = glob(str(ds_dir / "*_*_base.jsonld"))

    type2args = {
        "not-distinct": [
            queryGraphLocations, QueryPostprocessorNotDistinct, {}, CurrentEventsDataset,
        ],
        "distinct": [
            queryGraphLocations, QueryPostprocessorDistinct, {}, CurrentEventsDataset,
        ],
        "single-leaf-location": [
            queryGraphLocations, QueryPostprocessorSingleLeafLocation, {}, CurrentEventsDataset,
        ],
        # "entity-linking-locations": [
        #     queryGraphLocations, QueryPostprocessorSingleLeafLocationEntityLinking,
        #     {"loc2entity": get_loc2entity(ds_filepaths, basedir, forceExeptQuery or force)},
        #     CurrentEventsDataset,
        # ],
        "entity-linking": [
            queryGraphEntitys, QueryPostprocessorEntityLinking, {}, CurrentEventsDatasetEL,
        ],
        "entity-linking-wd": [
            queryGraphEntitys, QueryPostprocessorEntityLinkingWikidata, {}, CurrentEventsDataset,
        ],
        "entity-linking-untokenized": [
            queryGraphEntitys, QueryPostprocessorEntityLinkingUntokenized, {}, CurrentEventsDatasetRaw,
        ],
        "entity-linking-untokenized-wd": [
            queryGraphEntitys, QueryPostprocessorEntityLinkingUntokenizedWikidata, {}, CurrentEventsDatasetRaw,
        ],
        "entity-linking-untokenized-title": [
            queryGraphEntitys, QueryPostprocessorEntityLinkingUntokenizedTitle, {}, CurrentEventsDatasetRaw,
        ],
        "wikiurl2title": [
            queryGraphEntitys, None, {}, CurrentEventsDatasetWikiUrl2Title,
        ],
    }

    queryFunction = type2args[ds_type][0]
    queryPostprocessor = type2args[ds_type][1]
    qp_kwargs = type2args[ds_type][2]
    ds_class = type2args[ds_type][3]

    # create json dataset
    json_ds_path = createJsonDataset(ds_type, ds_dir, queryFunction, queryPostprocessor, 
            qp_kwargs, num_processes, forceExeptQuery, force)

    # create finished dataset
    if isinstance(ds_class, (CurrentEventsDataset, CurrentEventsDatasetEL)):
        ds = ds_class(json_ds_path, tokenizer)
    else:
        ds = ds_class(json_ds_path)

    return ds

