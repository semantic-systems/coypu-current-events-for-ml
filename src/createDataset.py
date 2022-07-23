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
from .queryPostprocessor import (QueryPostprocessorDistinct,
                                 QueryPostprocessorNotDistinct)


def createDataset(tokenizer, ds_dir:Path, ds_type:str, num_processes:int, force=False):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)

        all_labels = examples["labels"]
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
    
    output_ds_path = Path("./dataset/")
    
    type2qpp = {
        "not-distinct": QueryPostprocessorNotDistinct, 
        "distinct": QueryPostprocessorDistinct, 
    }

    if not exists(output_ds_path / ds_type / "train.json") or force: # only test for one
        # import json ds
        dataset_file_paths = graph2json_mp_host(ds_dir, type2qpp[ds_type], num_processes, force=force)

        #label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        #label_names = ['O', 'B-LOC', 'I-LOC']
        # features = Features({
        #     "tokens": Sequence(feature=Value(dtype='string')),
        #     "labels": Sequence(feature=ClassLabel(num_classes=len(label_names), names=label_names))
        # })
        ds = load_dataset('json', data_files=dataset_file_paths, field="data") # , features=features

        # tokenize data
        ds = ds.map(tokenize_and_align_labels, batched=True, remove_columns="tokens")

        print("Pre-split dataset:", ds)

        # split data
        test_size = 0.1
        val_size = 0.1
        ds = ds["train"].train_test_split(test_size=val_size) #, shuffle=True
        ds_val = ds["test"]
        ds = ds["train"].train_test_split(test_size=test_size*(1 / 1-val_size)) 
        ds_test = ds["test"]
        ds_train = ds["train"]

        # write
        makedirs(str(output_ds_path / ds_type), exist_ok=True)
        ds_val.to_json(str(output_ds_path / ds_type / "validation.json"))
        ds_test.to_json(str(output_ds_path / ds_type / "test.json"))
        ds_train.to_json(str(output_ds_path / ds_type / "train.json"))



def getTorchDataset(ds_type):
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
    
    return DatasetDict({
        "train" : CurrentEventsDataset(ds_type + "/train.json"),
        "validation" : CurrentEventsDataset(ds_type + "/validation.json"),
        "test" : CurrentEventsDataset(ds_type + "/test.json"),
    })
    
