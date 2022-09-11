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
from .queryPostprocessor import (
    QueryPostprocessorDistinct, QueryPostprocessorNotDistinct,
    QueryPostprocessorEntityLinking, QueryPostprocessorEntityLinkingWikidata,
    QueryPostprocessorSingleLeafLocation, QueryPostprocessorSingleLeafLocationEntityLinking,
    QueryPostprocessorEntityLinkingUntokenized, QueryPostprocessorEntityLinkingUntokenizedWikidata)

type2qpp = {
    "not-distinct": QueryPostprocessorNotDistinct, 
    "distinct": QueryPostprocessorDistinct, 
    "single-leaf-location": QueryPostprocessorSingleLeafLocation, 
    "entity-linking-locations": QueryPostprocessorSingleLeafLocationEntityLinking,
    "entity-linking": QueryPostprocessorEntityLinking,
    "entity-linking-wd": QueryPostprocessorEntityLinkingWikidata,
    "entity-linking-untokenized": QueryPostprocessorEntityLinkingUntokenized,
    "entity-linking-untokenized-wd": QueryPostprocessorEntityLinkingUntokenizedWikidata,
}

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

def getDataset(basedir, tokenizer, ds_dir:Path, ds_type:str, num_processes:int, forceExeptQuery=False, force=False, raw=False) -> Dataset:
    out_path = basedir / f"dataset/{ds_type}.json"
    ds_filepaths = glob(str(ds_dir / "*_*_base.jsonld"))
    qp_kwargs = {}

    if ds_type == "entity-linking-location":
        qp_kwargs["loc2entity"] = get_loc2entity(ds_filepaths, basedir, forceExeptQuery or force)

    
    if ds_type in ["entity-linking", "entity-linking-wd",
            "entity-linking-untokenized", "entity-linking-untokenized-wd"]:
        queryFunction = queryGraphEntitys
    else:
        queryFunction = queryGraphLocations

    if not exists(out_path) or forceExeptQuery or force: # only test for one
        # import json ds
        dataset_file_paths = graph2json_mp_host(
            ds_dir,
            ds_filepaths,
            type2qpp[ds_type], 
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
    
    # create finished dataset
    if raw: 
        ds = CurrentEventsDatasetRaw(ds_type + ".json")
    elif ds_type == "entity-linking":
        ds = CurrentEventsDatasetEL(ds_type + ".json", tokenizer)
    else:
        ds = CurrentEventsDataset(ds_type + ".json", tokenizer)

    return ds

class CurrentEventsDataset(Dataset):
    def __init__(self, filename, tokenizer):
        with open("dataset/" + filename, "r") as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        df = DataFrame.from_records(data)

        # tokenize data
        ds = hfDataset.from_pandas(df)
        ds = ds.map(tokenize_and_align_labels, batched=True, remove_columns="tokens", 
            fn_kwargs={"tokenizer":tokenizer, "label_key": "labels"})
        df = ds.to_pandas()

        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return {
            "labels": self.df.iloc[idx]["labels"], 
            "input_ids": self.df.iloc[idx]["input_ids"], 
            "attention_mask": self.df.iloc[idx]["attention_mask"]
        }
    def __repr__(self):
        return "CurrentEventsDataset: len=" + str(self.__len__())



class CurrentEventsDatasetEL(CurrentEventsDataset):
    def __init__(self, filename, tokenizer):
        with open("dataset/" + filename, "r") as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        df = DataFrame.from_records(data)
        print("Raw:")
        print(df.iloc[0])

        # convert entitys to ids
        entity2id, entity_list = self.__get_entity2id(df)
        df["labels"] = df["labels"].apply(lambda labels: [entity2id[l] for l in labels])
        print("entity2id:")
        print(df.iloc[0])

        # tokenize data
        ds = hfDataset.from_pandas(df)
        ds = ds.map(tokenize_and_align_labels, batched=True, remove_columns="tokens", 
            fn_kwargs={"tokenizer":tokenizer, "label_key": "labels"})
        df = ds.to_pandas()
        print("tokenized&aligned:")
        print(df.iloc[0])

        self.df = df
        self.entity_list = entity_list


    def __get_entity2id(self, df:DataFrame) -> Tuple[Dict, List]:
        entitys = set()
        for labels in df["labels"]:
            for label in labels:
                if label != "NIL":
                    entitys.add(label)

        entity_list_no_NIL = list(entitys)
        entity_list_no_NIL.sort()

        entity_list = ["NIL"]
        entity_list.extend(entity_list_no_NIL)

        entity2id = {e:i for i,e in enumerate(entity_list)}
        return entity2id, entity_list


class CurrentEventsDatasetRaw(Dataset):
    def __init__(self, filename):
        with open("dataset/" + filename, "r") as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        df = DataFrame.from_records(data)
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return {
            c: self.df.iloc[idx][c] for c in self.df.columns
        }

    def __repr__(self):
        return "CurrentEventsDataset: len=" + str(self.__len__())