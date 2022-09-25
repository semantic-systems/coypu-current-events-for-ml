from . import eval_module_dir
import argparse
from os.path import abspath, split
from pathlib import Path

from torch.utils.data import DataLoader

from ..datasets.currenteventstokg import currenteventstokg_module_dir
from ..datasets import datasets_module_dir
from ..datasets.createDataset import getDataset
from .eval_blink import eval_blink
from .eval_elq import eval_elq


if __name__ == '__main__':
    basedir, _ = split(abspath(__file__))
    basedir = Path(basedir)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # store_true
    parser.add_argument("-feq", '--force_exept_query', action='store_true',
        help="Ignore all caches exept query cache when creating dataset.")

    parser.add_argument("-f", '--force', action='store_true',
        help="Ignore all caches when creating dataset.")

    # store
    parser.add_argument('--kg_cache_dir', action='store', type=str, 
        default=str(currenteventstokg_module_dir / "cache/"),
        help="Directory of dataset generation cache.")
    
    parser.add_argument('--kg_ds_dir', action='store', type=str, 
        default=str(currenteventstokg_module_dir / "dataset/"),
        help="Directory of knowledge graph dataset.")
    
    parser.add_argument('--dataset_cache_dir', action='store', type=str, 
        default=str(datasets_module_dir / "dataset/"),
        help="Directory for datasets.")
    
    parser.add_argument('--cache_dir', action='store', type=str, 
        default=str(datasets_module_dir / "cache/"),
        help="Directory for module local caching.")
    
    parser.add_argument("-dt", '--ds_type', action='store', type=str, default="distinct") #, choices=

    parser.add_argument("-np", '--num_processes', action='store', type=int, default=4)

                        
    parser.add_argument("-m", "--model", action='store', type=str, default="blink",
        choices=["blink", "elq"],
        help="Evaluate entity linking models with dataset.")
    
    args = parser.parse_args()

    print(f"Evaluate {args.model} with ds")
    if args.model == "blink":
        eval_blink(basedir, args)
        pass
    elif args.model == "elq":
        eval_elq(basedir, args)
        pass
    