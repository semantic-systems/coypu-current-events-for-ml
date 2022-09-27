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
    #ds_filepaths = glob(str(ds_dir / "January_2020_base.jsonld"))
    #ds_filepaths.extend(glob(str(ds_dir / "January_2021_base.jsonld")))
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
    out_file_paths = []
    for f in kg_paths_chunk:
        filename = Path(f).parts[-1]
        splitted = filename.split("_")
        prefix = "_".join(splitted[0:2])

        if queryPostprocessor:
            suffix = queryPostprocessor.suffix

            out_file_name = prefix + suffix + ".json" # eg May_2020_TC_D.json
            out_file_path = str(ds_dir / out_file_name)

        if queryPostprocessor and exists(out_file_path) and not forceExeptQuery and not force:
            print(current_process().name, "File " + out_file_path + " exists.")    
        else:
            base_query_dir = ds_dir / str(queryFunc.__name__)
            base_query_file_path = str(base_query_dir / (prefix + ".json"))

            # query or use cache
            if exists(base_query_file_path) and not force:
                print(current_process().name, "Load cached query result from" + base_query_file_path + "...")
                with open(base_query_file_path, mode='r', encoding="utf-8") as f:
                    qres = load(f)
            else:
                g = Graph()

                n = Namespace("https://schema.coypu.org/global#")
                NIF = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
                # SEM = Namespace("http://semanticweb.cs.vu.nl/2009/11/sem/")
                # WGS = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")

                g.namespace_manager.bind('n', n)
                g.namespace_manager.bind('nif', NIF)

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

    q = """#PREFIX n: <http://data.coypu.org/>
SELECT DISTINCT ?text ?l_loc ?s_begin ?l_loc_begin ?l_loc_end WHERE{
    ?e rdf:type n:Event.
    ?e nif:isString ?text.
    ?e n:hasSentence ?s.

    ?s n:hasLink ?l;
        nif:beginIndex ?s_begin.

    ?l n:text ?l_loc;
        nif:beginIndex ?l_loc_begin;
        nif:endIndex ?l_loc_end;
        n:references ?a.

    ?a rdf:type n:Location.
    ?a owl:sameAs ?wd_a.
    
     FILTER NOT EXISTS { 
        ?e n:hasSentence ?s2.
        ?s2 n:hasLink ?l2.
        ?l2 n:references ?a2.
        ?a2 rdf:type n:Location.
        ?a2 owl:sameAs ?wd_cl.
        ?wd_cl n:hasParentLocation ?cl.
        ?cl n:parentLocation ?wd_a.
    } .
}"""
    res = g.query(q)

    rows = {}
    rows["data"] = []
    for row in res:
        # print(row.text.strip(" "), "\nl_loc:", row.l_loc)
        # #print(row.wd_pa, row.pa_loc, row.pwdl, row.plop, "\n")
        # print("\n")
        rows["data"].append({
            "text": row.text, "s_begin": row.s_begin, "location": row.l_loc, 
            "begin": row.l_loc_begin, "end": row.l_loc_end
        })
    
    return rows




def queryGraphEntitys(g:Graph) -> Dict[str,List]:
    print(current_process().name, "Running query ...")

    q = """SELECT DISTINCT ?text ?linktext ?s_begin ?begin ?end ?a ?wd ?title WHERE{
    ?e rdf:type n:Event.
    ?e nif:isString ?text.
    ?e n:hasSentence ?s.

    ?s n:hasLink ?l;
        nif:beginIndex ?s_begin.

    ?l n:hasText ?linktext;
        nif:beginIndex ?begin;
        nif:endIndex ?end;
        n:hasReference ?a.
    ?a owl:sameAs ?wd;
        n:hasName ?title.

}"""
    res = g.query(q)

    rows = {}
    rows["data"] = []
    for row in res:
        rows["data"].append({
            "text": row.text, "linktext": row.linktext, "s_begin": row.s_begin, 
            "begin": row.begin, "end": row.end, "article": row.a, "wd_entity": row.wd,
            "title": row.title
        })
    
    return rows
