from os import makedirs
from pathlib import Path
from rdflib import RDF, RDFS, XSD, BNode, Graph, Literal, Namespace, URIRef
from typing import Dict, List, Union
from json import dump, load


def get_loc2entity(ds_filepaths, basedir:Path, force:bool) -> Dict[str,str]:
    makedirs(basedir / "cache/", exist_ok=True)
    loc2entity_path = basedir / "cache/loc2entity.json"

    if exists(loc2entity_path) and not force:
        loc2entity = load_loc2entity(basedir)
    else:  
        print("Creating loc2entity...")

        q = """SELECT DISTINCT ?loc ?wd_a WHERE{
        ?l n:text ?loc;
            n:references ?a.

        ?a rdf:type n:Location;
            owl:sameAs ?wd_a.
    }"""
        loc2entity = {}

        g = Graph()

        n = Namespace("http://data.coypu.org/")
        g.namespace_manager.bind('n', n)

        for f in ds_filepaths:
            print("Parsing " + f + "...")
            g.parse(f)
        
        print("Querying " + f + "...")
        res = g.query(q)
        for row in res:
            loc2entity[str(row.loc)] = str(row.wd_a).split("/")[-1]
        
        # cache
        print(f"Caching loc2entity to {loc2entity_path}...")
        with open(loc2entity_path, mode='w', encoding="utf-8") as f:
            dump(loc2entity, f, separators=(',', ':'))

    return loc2entity

def load_loc2entity(basedir:Path):
    loc2entity_path = basedir / "cache/loc2entity.json"

    print(f"Load loc2entity from {loc2entity_path}...")
    with open(loc2entity_path, mode='r', encoding="utf-8") as f:
        loc2entity = load(f)
    return loc2entity

def create_entity_list(loc2entity:Dict) -> List:
    entity_set = { e for l,e in loc2entity.items() }
    print(f"entity_set len {len(entity_set)}")

    entity_list = list(entity_set)
    entity_list.sort()
    return entity_list