from json import dump, load
from multiprocessing import Pool, current_process
from os import makedirs
from os.path import exists
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Union

from rdflib import RDF, RDFS, XSD, BNode, Graph, Literal, Namespace, URIRef

from .queryPostprocessor import *

includedGraphExtensions = [] # "ohg", "osm", "raw"

def graph2json_mp_host(ds_dir, ds_filepaths, queryPostprocessor:QueryPostprocessor, queryFunc, 
        num_processes=4, forceExeptQuery=False, force=False, qp_kwargs={}) -> List[str]:
    print("Found these base graph files:")
    pprint(ds_filepaths)

    # parallelize
    def chunks(lst, n):
        res = []
        from math import ceil
        step = ceil(len(lst)/n)
        for i in range(0, (n-1)*step, step):
            res.append(lst[i: i+step])
        res.append(lst[(n-1)*step: len(lst)])
        return res

    filepaths_chunks = chunks(ds_filepaths, num_processes)
    print("Number of Workers:", num_processes)
    for i, c in enumerate(filepaths_chunks):
        print("Chunk", i)
        for fp in c:
            print("", fp)
        
    filepaths_with_args = [(fpc, queryPostprocessor, ds_dir, forceExeptQuery, force, qp_kwargs, queryFunc) for fpc in filepaths_chunks]
    
    with Pool(num_processes) as p:
        out_paths = p.starmap(mp_worker, filepaths_with_args)
    
    # flatten result
    res = []
    for l in out_paths:
        res.extend(l)

    return res





    
def mp_worker(kg_paths_chunk, queryPostprocessor, ds_dir, forceExeptQuery, force, qp_kwargs, queryFunc):
    queryFunc_name = str(queryFunc.__name__)
    base_query_dir = ds_dir / queryFunc_name

    out_file_paths = []
    for f in kg_paths_chunk:
        filename = Path(f).parts[-1]
        splitted = filename.split("_")
        prefix = "_".join(splitted[0:2])

        if queryPostprocessor:
            suffix = queryPostprocessor.suffix

            out_file_name = prefix + suffix + ".json" # eg May_2020_TC_D.json
            out_file_dir = base_query_dir / f"QPP{suffix}"
            makedirs(out_file_dir, exist_ok=True)
            out_file_path = str(out_file_dir / out_file_name)

        if queryPostprocessor and exists(out_file_path) and not forceExeptQuery and not force:
            print(current_process().name, "File " + out_file_path + " exists.")
        else:
            
            base_query_file_path = str(base_query_dir / (prefix + ".json"))

            # query or use cache
            if exists(base_query_file_path) and not force:
                print(current_process().name, "Load cached query result from" + base_query_file_path + "...")
                with open(base_query_file_path, mode='r', encoding="utf-8") as f:
                    qres = load(f)
            else:
                g = Graph()

                NIF = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
                COY = Namespace("https://schema.coypu.org/global#")
                WGS = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
                GEO = Namespace("http://www.opengis.net/ont/geosparql#")
                WD = Namespace("http://www.wikidata.org/entity/")
                CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
                GN = Namespace("https://www.geonames.org/ontology#")
                SCHEMA = Namespace("https://schema.org/")
                DCTERMS = Namespace("http://purl.org/dc/terms/")

                g.namespace_manager.bind('nif', NIF)
                g.namespace_manager.bind('coy', COY)
                g.namespace_manager.bind('wgs', WGS)
                g.namespace_manager.bind('geo', GEO)
                g.namespace_manager.bind('wd', WD)
                g.namespace_manager.bind('crm', CRM)
                g.namespace_manager.bind('gn', GN)
                g.namespace_manager.bind('schema', SCHEMA)
                g.namespace_manager.bind('dcterms', DCTERMS)

                # parse basegraph
                print(current_process().name, "Parsing " + f + "...")
                g.parse(f)

                # parse extensions of basegraph
                for e in includedGraphExtensions:
                    eName = prefix + "_" + e + ".jsonld"
                    ePath = str(ds_dir / eName)
                    print(current_process().name, "Parsing " + ePath + "...")
                    g.parse(ePath)
                
                qres = queryFunc(g)

                # cache base query 
                print(current_process().name, "Cache query to " + base_query_file_path + "...")
                makedirs(base_query_dir, exist_ok=True)
                with open(base_query_file_path, mode='w', encoding="utf-8") as f:
                    dump(qres, f, separators=(',', ':'))
            
            # query and covert
            if queryPostprocessor:
                res = queryPostprocessor.postprocess(qres, **qp_kwargs)

                # save 
                print(current_process().name, "Dump JSON to " + out_file_path + "...")
                with open(out_file_path, mode='w', encoding="utf-8") as f:
                    dump(res, f, separators=(',', ':'))
            else:
                res = qres
                out_file_path = base_query_file_path

        out_file_paths.append(out_file_path)
    return out_file_paths

    

def queryGraphLocations(g:Graph) -> Dict[str,List]:
    print(current_process().name, "Running query ...")

    q = """SELECT DISTINCT ?text ?l_loc ?s_begin ?l_loc_begin ?l_loc_end WHERE{
    ?e  a nif:Context;
        nif:isString ?text;
        nif:subString ?s.

    ?s  nif:subString ?l;
        nif:beginIndex ?s_begin.

    ?l  nif:anchorOf ?l_loc;
        nif:beginIndex ?l_loc_begin;
        nif:endIndex ?l_loc_end;
        gn:wikipediaArticle ?a.

    ?a  owl:sameAs ?wd_a.

    ?p  a coy:Location;
        gn:wikipediaArticle ?a.
    
     FILTER NOT EXISTS { 
        ?e nif:subString ?s2.
        ?s2 nif:subString ?l2.
        ?l2 gn:wikipediaArticle ?a2.
        ?p2 a coy:Location;
            gn:wikipediaArticle ?a2.
        ?p2 coy:isLocatedIn ?p.
    } .
}"""
    res = g.query(q)

    rows = {}
    rows["data"] = []
    for row in res:
        rows["data"].append({
            "text": row.text, "s_begin": row.s_begin, "location": row.l_loc, 
            "begin": row.l_loc_begin, "end": row.l_loc_end
        })
    
    return rows




def queryGraphEntitys(g:Graph) -> Dict[str,List]:
    print(current_process().name, "Running query ...")

    q = """SELECT DISTINCT ?text ?linktext ?s_begin ?begin ?end ?a_url ?wd ?title WHERE{
    ?e  a nif:Context;
        nif:isString ?text;
        nif:subString ?s.

    ?s  a nif:Sentence;
        nif:subString ?l;
        nif:beginIndex ?s_begin.

    ?l  a nif:Phrase;
        nif:anchorOf ?linktext;
        nif:beginIndex ?begin;
        nif:endIndex ?end;
        gn:wikipediaArticle ?a.
    
    ?a  owl:sameAs ?wd;
        schema:name ?title;
        dcterms:source ?a_url.
}"""
    res = g.query(q)

    rows = {}
    rows["data"] = []
    for row in res:
        rows["data"].append({
            "text": row.text, "linktext": row.linktext, "s_begin": row.s_begin, 
            "begin": row.begin, "end": row.end, "article": row.a_url, "wd_entity": row.wd,
            "title": row.title
        })
    
    return rows
