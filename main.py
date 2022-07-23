import argparse
from pathlib import Path
from pprint import pprint

import torch
from accelerate import Accelerator

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, get_scheduler)

from src.createDataset import CurrentEventsDataset, createDataset, type2qpp, tokenize_and_align_labels
from src.metrics import calculate_metrics, postprocess
from src.eval_conll2003 import eval_conll2003

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # store_true
    parser.add_argument("-cd", '--create_dataset', action='store_true',
        help="Create dataset for ml from knowledge graph dataset and exit.")

    parser.add_argument("-f", '--force', action='store_true',
        help="Ignore cache when creating dataset.")
 
    parser.add_argument("--eval_conll", action='store_true',
        help="Evaluate with conll2003.")

    # store
    parser.add_argument('--kg_ds_dir', action='store', help="Directory of knowledge graph dataset.",
                        default="../current-events-to-kg/dataset/")
    
    parser.add_argument('--ds_type', action='store', type=str, default="distinct", choices=list(type2qpp.keys()))

    parser.add_argument('--num_processes', action='store', type=int, default=4)

    parser.add_argument('--model_checkpoint', action='store', default="distilbert-base-uncased")
    
    parser.add_argument('--batch_size', action='store', type=int, default=16)
    
    parser.add_argument('--shuffle', action='store', type=bool, default=True)
    
    parser.add_argument('--num_train_epochs', action='store', type=int, default=4)
    
    parser.add_argument('--learning_rate', action='store', type=float, default=3e-5)

    parser.add_argument('--eval_steps', action='store', type=int, default=100)
                        
    
    args = parser.parse_args()

    # parameters
    model_checkpoint = args.model_checkpoint
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_train_epochs = args.num_train_epochs

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    if args.create_dataset:
        createDataset(tokenizer, Path(args.kg_ds_dir), args.ds_type, args.num_processes, args.force)
        print("Done!")
        quit()

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # load model
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
    
    # load stuff
    model = model_init()
    
    if args.eval_conll:
        # evaluate model
        eval_conll2003(model, data_collator, tokenizer, batch_size, id2label)

    else:
        # training

        # load dataset
        ds = CurrentEventsDataset(args.ds_type + ".json")
        print("Dataset:", ds)

        # split data
        ds_size = len(ds)
        test_size = int(0.1 * ds_size)
        val_size = int(0.1 * ds_size)
        train_size = ds_size - test_size - val_size    

        ds_train, ds_val, ds_test = random_split(ds, [train_size, val_size, test_size])
        print("ds_train:", len(ds_train))
        print("ds_val:", len(ds_val))
        print("ds_test:", len(ds_test))

        # print(ds["train"][0])
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

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
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



# def metrics(self, pred, labels, mode, epoch):
    #     def calculate_metrics(pred_tags, gt_tags):
    #         f1 = f1_score(pred_tags, gt_tags)*100
    #         ppv = precision_score(pred_tags, gt_tags)*100
    #         sen = recall_score(pred_tags, gt_tags)*100
    #         acc = accuracy_score(pred_tags, gt_tags)*100
    #         return {'f1':f1, 'precision':ppv, 'recall':sen, 'accuracy':acc}
        
    #     # get metrics on all labels
    #     metric = {}
    #     metric['all'] = calculate_metrics(pred, labels)
    #     for m in ['accuracy', 'f1', 'precision', 'recall']:
    #         self.tsboard[mode].add_scalar('metrics/{}'.format(m), metric['all'][m], epoch)
            
    #     # get metrics on single label
    #     metric['individual'] = {}
    #     for tag in self.label2id.keys():
    #         if tag != 'O' and tag != self.pad_token and tag in labels:
    #             pred = [p for p, g in zip(pred, labels) if p==tag or g==tag]
    #             gt = [g for p, g in zip(pred, labels) if p==tag or g==tag]
    #             metric['individual'][tag] = calculate_metrics(pred, gt)
            
    #     return metric
