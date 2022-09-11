import pandas as pd
import numpy as np 
from time import sleep
from torch.utils.data import Dataset
from pathlib import Path

class AidaDataset(Dataset):
    def __init__(self, basedir:Path, path:str="./aida-yago2-dataset/AIDA-YAGO2-dataset.tsv"):
        sentences, links_list = self.__load_aida(basedir, path)

        d = {
            "tokens":sentences,
            "labels":links_list,
        }
        self.df = pd.DataFrame(data=d)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return {
            c: self.df.iloc[idx][c] for c in self.df.columns
        }

    def __repr__(self):
        return (f"AidaDataset: len={str(self.__len__())}\n" 
         + f"Columns: {list(self.df.columns)}")
    
    def __load_aida(self, basedir:Path, path:str):
        with open(basedir / path, "r", encoding="utf-8") as f:
            sentences = []
            links_list = []

            sentence = []
            links = []
            for i,l in enumerate(f):
                if i == 0:
                    continue

                tokens = l.strip("\n").split("\t", 7)
                if l == "\n":
                    sentences.append(sentence)
                    links_list.append(links)

                    sentence = []
                    links = []
                else:
                    a = ['NIL'] * 7
                    for i,t in enumerate(tokens):
                        a[i] = t
                    sentence.append(a[0])
                    links.append(a[4])
        
        return sentences, links_list


# dataset provided needs to have columns for "tokens" and "labels".
# the label for none needs to be "NIL".
def generate_data_to_link_blink_from_tokens(ds):
    data_to_link = []
    # iterate over sentences
    for x in ds:
        labels = x["labels"]
        tokens = x["tokens"]

        # iterate over mentions in the sentence
        mention = ""
        left_context = ""
        i = 0
        while i < len(labels):
            label = labels[i]

            if i < len(labels)-1:
                next_label = labels[i+1]
            else:
                next_label = "NIL"
            
            # add part of mention
            if label != "NIL":
                if mention != "":
                    mention += " "    
                mention += tokens[i]
            
            # check if this is last part of mention
            if label != next_label or i == len(labels)-1:

                # first save sentence with current mention
                if mention != "":
                    if i < len(labels)-1:
                        right_context = " ".join(tokens[i+1:])
                    else:
                        right_context = ""
                        
                    data_to_link.append({
                        "id": len(data_to_link),
                        "label": "unknown",
                        "label_id": -1,
                        "left_context":left_context,
                        "mention":mention,
                        "Wikipedia_URL": label,
                        "right_context":right_context,
                    })
                
                # then reset mention
                mention = ""
                left_context = " ".join(tokens[:i+1])
                      
            i += 1

    return data_to_link

# dataset provided needs to have columns for "text" and 
# "mentions" (array with start, end, mention, url).
def generate_data_to_link_blink(ds):
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
            data_to_link.append({
                "id": len(data_to_link),
                "label": "unknown",
                "label_id": -1,
                "left_context":text[:start-1] if start > 0 else "",
                "mention":text[start:end],
                "Wikipedia_URL": url,
                "right_context":text[end:],
            })
    return data_to_link


# dataset provided needs to have columns for "tokens" and "labels".
# the label for none needs to be "NIL".
def generate_data_to_link_elq_from_tokens(ds):
    data_to_link = []
    # iterate over sentences
    for x in ds:
        labels = x["labels"]
        tokens = x["tokens"]

        res_sentence = {
            "id": len(data_to_link),
            "text": "",
            "mentions":[],
            "test":[]
        }

        # iterate over mentions in the sentence
        current_mention_start = 0
        i = 0
        while i < len(labels):
            label = labels[i]
            token = tokens[i]

            if i < len(labels)-1:
                next_label = labels[i+1]
            else:
                next_label = "NIL"

            # add token to text
            if len(res_sentence["text"]) > 0:
                res_sentence["text"] += " "
            res_sentence["text"] += token.lower()
            
            # check if this is last part of mention
            if label != next_label or i == len(labels)-1:
                
                # first save current mention span
                if label != "NIL":
                    res_sentence["mentions"].append([current_mention_start, len(res_sentence["text"])])


                # then set next mention start
                current_mention_start = len(res_sentence["text"])+1
            i += 1
        
        data_to_link.append(res_sentence)

    return data_to_link

# dataset provided needs to have columns for "text" and 
# "mentions" (array with start, end, mention, url).
def generate_data_to_link_elq(ds):
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
            wde = m[3].split("/")[-1]
            data_to_link.append({
                "text": text,
                "mentions": [start, end],
                "wikidata_id": wde
            })
    return data_to_link


