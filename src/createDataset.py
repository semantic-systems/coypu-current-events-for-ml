from glob import glob
import json
from json import dump
from os import makedirs
from os.path import exists
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Union

from datasets import (ClassLabel, DatasetDict, Features, Sequence, Value,
                      load_dataset)
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from .graph2json import graph2json_mp_host
from .entity_linking import get_loc2entity
from .queryPostprocessor import (QueryPostprocessorDistinct,
                                 QueryPostprocessorNotDistinct, 
                                 QueryPostprocessorSingleLeafLocation,
                                 QueryPostprocessorSingleLeafLocationEntityLinking)

type2qpp = {
    "not-distinct": QueryPostprocessorNotDistinct, 
    "distinct": QueryPostprocessorDistinct, 
    "single-leaf-location": QueryPostprocessorSingleLeafLocation, 
    "entity-linking": QueryPostprocessorSingleLeafLocationEntityLinking
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

def createDataset(basedir, tokenizer, ds_dir:Path, ds_type:str, num_processes:int, forceExeptQuery=False, force=False):
    out_path = basedir / f"dataset/{ds_type}.json"
    ds_filepaths = glob(str(ds_dir / "*_*_base.jsonld"))
    qp_kwargs = {}

    if ds_type == "entity-linking":
        qp_kwargs["loc2entity"] = get_loc2entity(ds_filepaths, basedir, forceExeptQuery or force)

    if not exists(out_path) or forceExeptQuery or force: # only test for one
        # import json ds
        dataset_file_paths = graph2json_mp_host(
            ds_dir,
            ds_filepaths,
            type2qpp[ds_type], 
            num_processes, 
            forceExeptQuery=forceExeptQuery, 
            force=force, 
            qp_kwargs=qp_kwargs
        )

        #label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        #label_names = ['O', 'B-LOC', 'I-LOC']
        # features = Features({
        #     "tokens": Sequence(feature=Value(dtype='string')),
        #     "labels": Sequence(feature=ClassLabel(num_classes=len(label_names), names=label_names))
        # })
        ds = load_dataset('json', data_files=dataset_file_paths, field="data") # , features=features
        print(ds["train"][0])

        # tokenize data
        ds = ds.map(tokenize_and_align_labels, batched=True, remove_columns="tokens", 
            fn_kwargs={"tokenizer":tokenizer, "label_key": "labels"})

        print("Dataset:", ds)
        print(f"Save dataset to {out_path}")
        ds["train"].to_json(out_path)
    else:
        print(f"Datset already exists at {out_path}")


class CurrentEventsDataset(Dataset):
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
            "labels": self.df.iloc[idx]["labels"], 
            "input_ids": self.df.iloc[idx]["input_ids"], 
            "attention_mask": self.df.iloc[idx]["attention_mask"]
        }
    def __repr__(self):
        return "CurrentEventsDataset: len=" + str(self.__len__())
    
