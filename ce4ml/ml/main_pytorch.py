import argparse
from pathlib import Path
from pprint import pprint

from os import makedirs
import torch
from accelerate import Accelerator

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, get_scheduler)

from ..datasets.createDataset import CurrentEventsDataset, CurrentEventsDatasetEL, getDataset, tokenize_and_align_labels
from .metrics import calculate_metrics, postprocess
from .eval_conll2003 import eval_conll2003
from .entity_linking import load_loc2entity, create_entity_list
from ..datasets.currenteventstokg import currenteventstokg_dir
from ..datasets import datasets_module_dir

from os.path import abspath, split

class EarlyStopper:
    def __init__(self, num_unchanged_epochs=5):
        self.tolerance = num_unchanged_epochs
        self.stop = False
        self.epoch_counter = 0
        self.last_loss = None
    
    def stop_or_not(self, loss):
        if loss < self.last_loss:
            self.last_loss = loss
            self.epoch_counter = 0
        elif loss > self.min_validation_loss:
            self.epoch_counter += 1
            if self.epoch_counter >= self.epoch_counter:
                return True
        return False

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
    
    parser.add_argument('--num_train_epochs', action='store', type=int, default=4)

    parser.add_argument('--learning_rate', action='store', type=float, default=3e-5)

    parser.add_argument('--eval_steps', action='store', type=int, default=200)
    
    parser.add_argument('--warmup_steps', action='store', type=float, default=0,
        help="Percentage of total training steps used for warmup [0-1]")

    args = parser.parse_args()

    # parameters
    model_checkpoint = args.model_checkpoint
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_train_epochs = args.num_train_epochs

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # load model
    if args.ds_type == "entity-linking-location":
        loc2entity = load_loc2entity(basedir)
        label_names = ["NIL"]
        label_names.extend(create_entity_list(loc2entity))
    else:
        label_names = ['O', 'B-LOC', 'I-LOC']
    
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    # Function that returns an untrained model to be trained
    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )
    
    model = model_init()

    # load dataset
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

    
    if args.eval_conll:
        # evaluate model
        eval_conll2003(model, data_collator, tokenizer, batch_size, id2label)

    else:
        # training
        
        cache_dir = basedir / "cache" 
        makedirs(cache_dir)



        ## load data
        # split data
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

        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate
        )

        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch
        num_warmup_steps = int(num_training_steps * args.warmup_steps)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

         # fine tune model
        writer = SummaryWriter()
        for epoch in range(num_train_epochs):
            # Training
            metrics = {}
            for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                model.train()

                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if i > 0 and i % args.eval_steps == 0:
                    # Evaluation
                    model.eval()
                    true_predictions, true_labels = [], []
                    for batch in tqdm(eval_dataloader, desc="Evaluating"):
                        with torch.no_grad():
                            outputs = model(**batch)
                            eval_loss = outputs.loss

                        predictions = outputs.logits.argmax(dim=-1)
                        labels = batch["labels"]

                        # Necessary to pad predictions and labels for being gathered
                        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                        predictions_gathered = accelerator.gather(predictions)
                        labels_gathered = accelerator.gather(labels)

                        true_predictions_batch, true_labels_batch = postprocess(predictions_gathered, labels_gathered, id2label)
                        true_predictions.extend(true_predictions_batch)
                        true_labels.extend(true_labels_batch)
                    
                    metrics = calculate_metrics(true_predictions, true_labels)
                    
                    # log to tensorboard
                    global_time = epoch*len(train_dataloader)+i
                    writer.add_scalar('train/loss', loss, global_time)
                    writer.add_scalar('eval/loss', eval_loss, global_time)
                    for m in ['accuracy', 'f1', 'precision', 'recall']:
                        writer.add_scalar('eval/{}'.format(m), metrics[m], global_time)
                    print(
                        f"epoch {epoch}, step {i}:",
                        {
                            key: metrics[key]
                            for key in ["precision", "recall", "f1", "accuracy"]
                        },
                    )
        # save model
        output_dir = "model_checkpoint"
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

        # end of fine-tuning
