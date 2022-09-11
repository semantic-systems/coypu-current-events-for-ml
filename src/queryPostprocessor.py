from abc import ABC, abstractmethod
from typing import Dict
from pprint import pprint
from .entity_linking import create_entity_list

class QueryPostprocessor(ABC):
    @staticmethod
    @abstractmethod
    def postprocess(qres, **kwargs):
        pass

    def _tokenize_and_label_location_row(row):
        punctuations = [".", ",", ":", ";"]

        textTokens = []
        labels = []

        text = str(row["text"])
        sentence_begin = int(row["s_begin"])
        begin = int(row["begin"])
        end = int(row["end"])
        location = str(row["location"])

        true_begin = sentence_begin + begin
        true_end = sentence_begin + end

        tok = ""
        lastCharWasLoc = False
        for i in range(len(text)):
            char = text[i]
            if not char.isspace() and char not in punctuations and not (lastCharWasLoc and i >= end):
                # char is no end of (location) token, add char to current token
                tok += char
                lastCharWasLoc = (i >= true_begin and i < true_end)
            else:
                # token end -> append token and label
                textTokens.append(tok)
                tok = ""
                if lastCharWasLoc:
                    if len(labels) == 0 or labels[-1] == 0:
                        labels.append(1) # B-LOC
                    else:
                        labels.append(2) # I-LOC
                else:
                    labels.append(0)

                if char in punctuations:
                    textTokens.append(char)
                    labels.append(0)

                lastCharWasLoc = False
        
        assert len(labels) == len(textTokens)
        assert 1 in labels or 2 in labels, {"row":row, "tokens":textTokens, "labels":labels}

        return {"tokens":textTokens, "labels":labels, "location": location}
    
    def _tokenize_and_label_entity_linking(rowList, label_key="article"):
        punctuations = [".", ",", ":", ";"]
        nilLabel = "NIL"
        textTokens = []
        labels = []

        text = str(rowList[0]["text"])

        links=[]
        for row in rowList:
            links.append((
                str(row[label_key]),
                int(row["begin"]) + int(row["s_begin"]), 
                int(row["end"]) + int(row["s_begin"])
            ))

        # split text into tokens and label them
        tok = ""
        for i,char in enumerate(text):
            
            # check if char is on boundary or inside of a link
            link_begins_next = False
            link_ended = False
            label = "NIL"
            for entity, begin, end in links:
                link_begins_next |= (i+1 == begin)
                link_ended |= (i == end)
                if i >= begin and i <= end:
                    label = entity

            if link_begins_next or link_ended or char.isspace() or char in punctuations:
                # end of token
                if link_begins_next:
                    tok += char

                tok_candidate = tok.strip()
                if len(tok_candidate) >= 1:
                    textTokens.append(tok_candidate)
                    labels.append(label)

                if char in punctuations:
                    # own token for punctuations
                    textTokens.append(char)
                    labels.append(nilLabel)

                tok = ""

                if link_ended and char not in punctuations:
                    tok += char
            else:
                # char is mid-token
                tok += char
        
        assert len(labels) == len(textTokens)
            
        return {"tokens":textTokens, "labels":labels}


class QueryPostprocessorNotDistinct(QueryPostprocessor):
    suffix = "_TC"
    @staticmethod
    def postprocess(inData:Dict, **kwargs):
        res = {"data":[]}
        for row in inData["data"]:
            res_row = QueryPostprocessor._tokenize_and_label_location_row(row)
            res["data"].append(res_row)
        return res

class QueryPostprocessorDistinct(QueryPostprocessor):
    suffix = "_TC_D"
    
    @staticmethod
    def postprocess(inData:Dict, **kwargs):
        res = {"data":[]}

        row_dict = {}
        for row in inData["data"]:
            if row["text"] in row_dict:
                row_dict[row["text"]].append(row)
            else:
                row_dict[row["text"]] = [row]
        
        for rowList in row_dict.values():
            res_row = QueryPostprocessor._tokenize_and_label_location_row(rowList[0])
            res["data"].append(res_row)
        print("Row count to postprocess:", len(inData["data"]), "resulting in:", len(res["data"]))
        return res

class QueryPostprocessorSingleLeafLocation(QueryPostprocessor):
    suffix = "_TC_SLL"

    @staticmethod
    def postprocess(inData:Dict, **kwargs):
        res = {"data":[]}

        row_dict = {}
        for row in inData["data"]:
            if row["text"] in row_dict:
                row_dict[row["text"]].append(row)
            else:
                row_dict[row["text"]] = [row]
        
        for rowList in row_dict.values():
            if len(rowList) == 1:
                res_row = QueryPostprocessor._tokenize_and_label_location_row(rowList[0])
                res["data"].append(res_row)
        print("Row count to postprocess:", len(inData["data"]), 
            "\ndistinct events:", len(row_dict.values()), 
            "\nevents with single leaf location:", len(res["data"])
        )
        return res

class QueryPostprocessorSingleLeafLocationEntityLinking(QueryPostprocessor):
    suffix = "_TC_LOC_EL"

    @staticmethod
    def postprocess(inData:Dict, loc2entity:Dict):
        res = {"data":[]}
        entity_list = create_entity_list(loc2entity)
        entity2id = {e:i for i,e in enumerate(entity_list, 1)}
        

        row_dict = {}
        for row in inData["data"]:
            if row["text"] in row_dict:
                row_dict[row["text"]].append(row)
            else:
                row_dict[row["text"]] = [row]
        
        for rowList in row_dict.values():
            if len(rowList) == 1:
                row = rowList[0]
                row = QueryPostprocessor._tokenize_and_label_location_row(row)
                entity = entity2id[loc2entity[ row["location"] ]]
                
                
                # change ner tags to entity
                row["labels"] = [
                    entity if l in [1,2] else 0 for l in row["labels"] 
                ]

                res["data"].append(row)
        print("Row count to postprocess:", len(inData["data"]), 
            "\ndistinct events:", len(row_dict.values()), 
            "\nevents with single leaf location (entity):", len(res["data"])
        )
        return res

class QueryPostprocessorEntityLinking(QueryPostprocessor):
    suffix = "_TC_EL"

    @staticmethod
    def postprocess(inData:Dict):
        return QueryPostprocessorEntityLinking._postprocess(inData, "article")

    @staticmethod
    def _postprocess(inData:Dict, label_key:str):
        res = {"data":[]}        

        row_dict = {}
        for row in inData["data"]:
            if row["text"] in row_dict:
                row_dict[row["text"]].append(row)
            else:
                row_dict[row["text"]] = [row]
        
        for rowList in row_dict.values():
            merged_row = {}
            for row in rowList:
                r = QueryPostprocessor._tokenize_and_label_entity_linking(rowList, label_key=label_key)
                
                if "tokens" not in merged_row:
                    merged_row["tokens"] = r["tokens"]

                if "labels" not in merged_row:
                    merged_row["labels"] = r["labels"]
                else:
                    assert len(merged_row["labels"]) == len(r["labels"])
                    for i,l in enumerate(r["labels"]):
                        if l != merged_row["labels"][i] and l != "NIL":
                            merged_row["labels"][i] = l
                
            res["data"].append(merged_row)
        print("Row count to postprocess:", len(inData["data"]), 
            "\ndistinct events:", len(row_dict.values()), 
            "\nevents with linked entity:", len(res["data"])
        )
        return res

class QueryPostprocessorEntityLinkingWikidata(QueryPostprocessorEntityLinking):
    suffix = "_TC_EL_WD"

    @staticmethod
    def postprocess(inData:Dict):
        return QueryPostprocessorEntityLinking._postprocess(inData, "wd_entity")

class QueryPostprocessorEntityLinkingUntokenized(QueryPostprocessor):
    suffix = "_EL"

    @staticmethod
    def postprocess(inData:Dict):
        return QueryPostprocessorEntityLinkingUntokenized._postprocess(inData, "article")

    @staticmethod
    def _postprocess(inData:Dict, label_key:str):
        res = {"data":[]}

        row_dict = {}
        for row in inData["data"]:
            if row["text"] in row_dict:
                row_dict[row["text"]].append(row)
            else:
                row_dict[row["text"]] = [row]
        
        for rowList in row_dict.values():
            merged_row = {
                "text": rowList[0]["text"],
                "mentions": [],
            }
            for row in rowList:
                begin = row["begin"] + row["s_begin"]
                end = row["end"] + row["s_begin"]
                mention = row["linktext"]
                entity = row[label_key]
                merged_row["mentions"].append((begin, end, mention, entity))                
                
            res["data"].append(merged_row)

        print("Row count to postprocess:", len(inData["data"]), 
            "\ndistinct events:", len(row_dict.values()), 
            "\nevents with linked entity:", len(res["data"])
        )
        return res

class QueryPostprocessorEntityLinkingUntokenizedWikidata(QueryPostprocessorEntityLinkingUntokenized):
    suffix = "_EL_WD"

    @staticmethod
    def postprocess(inData:Dict):
        return QueryPostprocessorEntityLinkingUntokenized._postprocess(inData, "wd_entity")
    