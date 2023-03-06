import argparse
from os import makedirs
from os.path import split, abspath
from pathlib import Path
from typing import List

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from .entity_linking import create_entity_list, load_loc2entity
from ..datasets import datasets_module_dir
from ..datasets.currenteventstokg import currenteventstokg_dir
from ..datasets.createDataset import getDataset
from torch.utils.data import DataLoader, random_split
from .pl_model import LocationExtractor


if __name__ == '__main__':
    basedir, _ = split(abspath(__file__))
    basedir = Path(basedir)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # store_true
    parser.add_argument("-feq", '--force_exept_query', action='store_true',
        help="Ignore all caches exept query cache when creating dataset.")

    parser.add_argument("-f", '--force', action='store_true',
        help="Ignore all caches when creating dataset.")
 
    parser.add_argument("--eval_conll", action='store_true',
        help="Evaluate with conll2003.")
    

    # store
    parser.add_argument('--dataset_cache_dir', action='store', type=str, 
        default=str(datasets_module_dir / "dataset/"),
        help="Directory for datasets.")

    parser.add_argument('--kg_ds_dir', action='store', type=str, 
        default=str(currenteventstokg_dir / "dataset/"),
        help="Relative directory of knowledge graph dataset.")
    
    parser.add_argument('--ds_type', action='store', type=str, default="distinct")

    parser.add_argument("-np", '--num_processes', action='store', type=int, default=4)

    parser.add_argument('--model_checkpoint', action='store', default="distilbert-base-uncased")
    
    parser.add_argument('--batch_size', action='store', type=int, default=16)
    
    parser.add_argument('--shuffle', action='store', type=bool, default=True)
    
    parser.add_argument('--learning_rate', action='store', type=float, default=3e-5)

    parser.add_argument('--eval_steps', action='store', type=int, default=200)
    
    parser.add_argument('--warmup_length', action='store', type=float, default=0.1)

    args = parser.parse_args()

    # parameters
    model_checkpoint = args.model_checkpoint
    batch_size = args.batch_size
    shuffle = args.shuffle
    learning_rate = args.learning_rate
    warmup_length = args.warmup_length 

    # determinism
    pl.seed_everything(42, workers=True)

    # load dataset
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    ds = getDataset(
        tokenizer,
        Path(args.dataset_cache_dir),
        Path(args.kg_ds_dir), 
        args.ds_type, 
        args.num_processes, 
        args.force_exept_query, 
        args.force
    )
    print("Dataset:", ds)
    print("First row:")
    print(ds[0])

    ds_size = len(ds)
    test_size = int(0.1 * ds_size)
    val_size = int(0.1 * ds_size)
    train_size = ds_size - test_size - val_size

    ds_train, ds_val, ds_test = random_split(ds, [train_size, val_size, test_size])
    print("ds_train:", len(ds_train))
    print("ds_val:", len(ds_val))
    print("ds_test:", len(ds_test))

    train_dataloader = DataLoader(
        ds_train,
        shuffle=shuffle,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        ds_val,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(
        ds_test,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    # load model
    if args.ds_type == "entity-linking-location":
        loc2entity = load_loc2entity(basedir)
        label_names = ["NIL"]
        label_names.extend(create_entity_list(loc2entity))
    else:
        label_names = ['O', 'B-LOC', 'I-LOC']
    
    model = LocationExtractor(
        model_name_or_path=model_checkpoint, 
        num_batches_per_epoch=len(train_dataloader), 
        label_names=label_names,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_length=warmup_length,
    )

    # init trainer
    monitor_value = "val_loss"
    monitor_mode = "min"

    early_stop_callback = EarlyStopping(
        monitor=monitor_value, 
        patience=5, 
        mode=monitor_mode,
    )
    model_checkpoint_callback = ModelCheckpoint(
        monitor=monitor_value,
        mode=monitor_mode,
        save_top_k=1
    )

    trainer_dir = basedir / "pl_trainer"
    makedirs(trainer_dir, exist_ok=True)

    trainer = pl.Trainer(
        # accelerator='gpu',
        # devices=[0],
        default_root_dir=trainer_dir,
        callbacks=[model_checkpoint_callback, early_stop_callback],
    )

    # train
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader
    )
    
    # load best model
    model = LocationExtractor.load_from_checkpoint(model_checkpoint_callback.best_model_path)

    #test
    test_res = trainer.test(
        model=model,
        dataloaders=test_dataloader
    )
    