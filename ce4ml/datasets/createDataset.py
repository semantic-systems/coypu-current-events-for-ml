import json
from glob import glob
from os import makedirs
from os.path import exists
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Union, Tuple

from datasets import ClassLabel
from datasets import DatasetDict, Features, Sequence, Value, load_dataset
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from ..ml.entity_linking import get_loc2entity
from .graph2json import (graph2json_mp_host, queryGraphEntitys,
                         queryGraphLocations)
from .queryPostprocessor import *
from .CurrentEventsDatasets import *
from . import datasets_module_dir


def createJsonDataset(ds_type:str, dataset_input_dir:Path, dataset_output_dir:Path, queryFunction, 
        queryPostprocessor, qp_kwargs, num_processes, forceExeptQuery, force) -> Path:

    out_path = dataset_output_dir / f"{ds_type}.json"
    ds_filepaths = glob(str(dataset_input_dir / "*_*_base.jsonld"))

    if not exists(out_path) or forceExeptQuery or force:
        # import json ds
        dataset_file_paths = graph2json_mp_host(
            dataset_input_dir,
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

def getDataset(tokenizer, dataset_output_dir:Path, dataset_input_dir:Path, ds_type:str, num_processes:int,
        forceExeptQuery=False, force=False) -> Dataset:
    
    print(f"getting dataset {ds_type}")

    ds_filepaths = glob(str(dataset_input_dir / "*_*_base.jsonld"))

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
    json_ds_path = createJsonDataset(ds_type, dataset_input_dir, dataset_output_dir, queryFunction, queryPostprocessor, 
            qp_kwargs, num_processes, forceExeptQuery, force)

    # create finished dataset
    if ds_class in (CurrentEventsDataset, CurrentEventsDatasetEL):
        ds = ds_class(json_ds_path, tokenizer)
    else:
        ds = ds_class(json_ds_path)

    return ds

