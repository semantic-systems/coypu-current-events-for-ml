from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame
from typing import Dict, List, Union, Tuple
from pathlib import Path
import json


class CurrentEventsDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r") as f:
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
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r") as f:
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
    def __init__(self, file_path):
        with open(file_path, "r") as f:
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


class CurrentEventsDatasetElqEval(Dataset):
    def __init__(self, basedir:Path, ds_dir:Path, force, forceExeptQuery, num_processes):
        ds_path = createJsonDataset("elq-eval", ds_dir, queryGraphEntitys, 
                QueryPostprocessorEntityLinkingUntokenized,
                {}, num_processes, forceExeptQuery, force)
        
        with open(ds_path, "r") as f:
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
        return  f"CurrentEventsDatasetElqEval: \n" +\
                f"len={str(self.__len__())}\n" +\
                f"columns={self.df.columns}"

class CurrentEventsDatasetWikiUrl2Title(Dataset):
    def __init__(self, file_path):        
        self.url2title = {}
        with open(file_path, "r") as f:
            for line in f:
                l = json.loads(line)
                url = l["article"]
                title = l["title"]
                assert  url in self.url2title and self.url2title[url] == title or \
                        url not in self.url2title
                self.url2title[url] = l["title"]

    def __len__(self):
        return len(self.url2title)

    def __getitem__(self, key):
        return self.url2title[key]
    
    def __contains__(self, key):
        return key in self.url2title

    def __repr__(self):
        return  f"WikiUrl2Title: \n" +\
                f"len={str(self.__len__())}\n"