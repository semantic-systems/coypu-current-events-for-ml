from abc import ABC, abstractmethod
from typing import Dict

class QueryPostprocessor(ABC):
    @staticmethod
    @abstractmethod
    def postprocess(qres):
        pass

    def _tokenize_and_label(row):
        punctuations = [".", ",", ":", ";"]

        textTokens = []
        labels = []

        text = str(row["text"])
        sentence_begin = int(row["s_begin"])
        begin = int(row["begin"])
        end = int(row["end"])

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

        return {"tokens":textTokens, "labels":labels}
    


class QueryPostprocessorNotDistinct(QueryPostprocessor):
    suffix = "_TC"
    @staticmethod
    def postprocess(inData:Dict):
        res = {"data":[]}
        for row in inData["data"]:
            res_row = QueryPostprocessor._tokenize_and_label(row)
            res["data"].append(res_row)
        return res

class QueryPostprocessorDistinct(QueryPostprocessor):
    suffix = "_TC_D"
    
    @staticmethod
    def postprocess(inData:Dict):
        res = {"data":[]}

        row_dict = {}
        for row in inData["data"]:
            if row["text"] in row_dict:
                row_dict[row["text"]].append(row)
            else:
                row_dict[row["text"]] = [row]
        
        for rowList in row_dict.values():
            res_row = QueryPostprocessor._tokenize_and_label(rowList[0])
            res["data"].append(res_row)
        print("Row count to postprocess:", len(inData["data"]), "resulting in:", len(res["data"]))
        return res

class QueryPostprocessorSingleLeafLocation(QueryPostprocessor):
    suffix = "_TC_SLL"

    @staticmethod
    def postprocess(inData:Dict):
        res = {"data":[]}

        row_dict = {}
        for row in inData["data"]:
            if row["text"] in row_dict:
                row_dict[row["text"]].append(row)
            else:
                row_dict[row["text"]] = [row]
        
        for rowList in row_dict.values():
            if len(rowList) == 1:
                res_row = QueryPostprocessor._tokenize_and_label(rowList[0])
                res["data"].append(res_row)
        print("Row count to postprocess:", len(inData["data"]), 
            "\ndistinct events:", len(row_dict.values()), 
            "\nevents with single leaf location:", len(res["data"])
        )
        return res