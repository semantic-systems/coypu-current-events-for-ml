from atexit import register
from pathlib import Path
from time import sleep
import json
import requests
from bs4 import BeautifulSoup
import time
import re
from pprint import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd
import wikipedia
from src.createDataset import getDataset
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Union
from os.path import exists


class WikipediaTitleCache():
    def __init__(self, basedir, ignore_cache=False):
        self.cache_id2title_path = basedir / "cache" / "pageid2title.json"
        self.cache_url2title_path = basedir / "cache" / "url2title.json"

        self.id2title = self.__loadJsonDict(self.cache_id2title_path, ignore_cache)
        self.url2title = self.__loadJsonDict(self.cache_url2title_path, ignore_cache)

        self.lastQuery = 0

        # save caches after termination
        register(self.__saveCaches)
    
    def getTitleById(self, pageid):
        if id in self.id2title:
            return self.id2title[pageid]
        else:
            title = wikipedia.page(pageid=pageid).title
            self.id2title[pageid] = title


    def getTitleByUrl(self, url):
        if url in self.url2title:
            return self.url2title[url]
        else:
            self.queryCooldown(1)
            page = requests.get(url)
            p = BeautifulSoup(page.text, 'lxml')
            articleGraphTag = p.find("script", attrs={"type": "application/ld+json"})
            pageGraph = json.loads(articleGraphTag.string)
            title = pageGraph["name"]
            self.url2title[url] = title
            return title
        
    
    def queryCooldown(self, sec):
        now = time.time()
        t = sec - (now - self.lastQuery)
        if(t < 0):
            t = 0
        time.sleep(t)
        self.lastQuery = time.time()
    
    def __saveCaches(self):
        self.__saveJsonDict(self.cache_id2title_path, self.id2title)
        self.__saveJsonDict(self.cache_url2title_path, self.url2title)

    def __loadJsonDict(self, file_path, ignore):
        if(exists(file_path) and not ignore):
            with open(file_path, mode='r', encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}
    
    def __saveJsonDict(self, file_path, dic):
        with open(file_path, mode='w', encoding="utf-8") as f:
            json.dump(dic, f)


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
        url2title = getDataset(
            basedir,
            None,
            basedir / args.kg_ds_dir,
            "wikiurl2title",
            args.num_processes,
            args.force_exept_query,
            args.force
        )

        with open(basedir / path, "r", encoding="utf-8") as f:
            sentences = []
            links_list = []
            page_ids_list = []

            sentence = []
            links = []
            page_ids = []
            for i,line in enumerate(f):
                if i == 0:
                    continue

                tokens = line.strip("\n").split("\t", 7)
                if line == "\n":
                    sentences.append(sentence)
                    links_list.append(links)
                    page_ids_list.append(page_ids)

                    sentence = []
                    links = []
                    page_ids = []
                else:
                    a = ['NIL'] * 7
                    for i,t in enumerate(tokens):
                        a[i] = t
                    sentence.append(a[0])
                    links.append(a[4])
                    page_ids.append(a[5])
        
        return sentences, links_list

class AidaDatasetTitles(Dataset):
    def __init__(self, basedir:Path, args, path:str="./aida-yago2-dataset/AIDA-YAGO2-dataset.tsv"):
        sentences, mentions_list = self.__load(basedir, args, path)

        d = {
            "text":sentences,
            "mentions":mentions,
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
    
    def __load(self, basedir:Path, args, path:str):
        url2title = getDataset(
            basedir,
            None,
            basedir / args.kg_ds_dir,
            "wikiurl2title",
            args.num_processes,
            args.force_exept_query,
            args.force
        )
        title_cache = WikipediaTitleCache(basedir)

        with open(basedir / path, "r", encoding="utf-8") as f:
            sentences = []
            mentions_list = []

            sentence = ""
            mentions = []

            last_link = 'NIL'
            last_link_start = 0

            sanity_check = set()

            for i,line in enumerate(f):
                if i == 0:
                    continue

                tokens = line.strip("\n").split("\t", 7)
                a = ['NIL'] * 7
                for i,t in enumerate(tokens):
                    a[i] = t
                token = a[0]
                link = a[4]
                page_id = a[5]
                if link != 'NIL':
                    sanity_check.add(link)
                    if link in url2title:
                        title = url2title[link]
                    else:
                        title = title_cache.getTitleByUrl(link)

                if last_link != 'NIL' and (last_link != link or line == "\n"):
                    # mention ended
                    mention = sentence[last_link_start:]
                    m = [last_link_start, len(sentence), mention, last_link, title]
                    mentions.append(m)
                    
                   
                if line == "\n":
                    assert { m[3] for m in mentions } == sanity_check
                    # next sentence
                    sentences.append(sentence)
                    mentions_list.append(mentions)

                    print("o", end="", flush=True)

                    sentence = ""
                    mentions = []
                    sanity_check = set()
                else:
                    # append token
                    if len(sentence) > 0:
                        sentence += " "
                    
                    if last_link != link:
                        # set new mention start
                        last_link_start = len(sentence)
                    
                    sentence += token

                last_link = link
        
        return sentences, mentions_list


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
                        "context_left":left_context,
                        "mention":mention,
                        "Wikipedia_URL": label,
                        "context_right":right_context,
                    })
                
                # then reset mention
                mention = ""
                left_context = " ".join(tokens[:i+1])
                      
            i += 1

    return data_to_link






##### ELQ

"""
Samples: list of examples, each of the form--

IF HAVE LABELS
{
    "id": "WebQTest-12",
    "text": "who is governor of ohio 2011?",
    "mentions": [[19, 23], [7, 15]],
    "tokenized_text_ids": [2040, 2003, 3099, 1997, 4058, 2249, 1029],
    "tokenized_mention_idxs": [[4, 5], [2, 3]],
    "label_id": [10902, 28422],
    "wikidata_id": ["Q1397", "Q132050"],
    "entity": ["Ohio", "Governor"],
    "label": [list of wikipedia descriptions]
}

IF NO LABELS (JUST PREDICTION)
{
    "id": "WebQTest-12",
    "text": "who is governor of ohio 2011?",
}
"""


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





