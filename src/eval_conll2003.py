import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from .createDataset import tokenize_and_align_labels
from .metrics import calculate_metrics, postprocess


def eval_conll2003(model, data_collator, tokenizer, batch_size, id2label):
    ds = load_dataset("conll2003")
    print(ds)

    # filter everything out exept location labels
    df = ds["test"].to_pandas()
    #print(df)
    def filterForLocation(x):
        return [ (tag - 4) if tag in [5,6] else 0 for tag in x ]
    df["ner_tags"] = df["ner_tags"].apply(filterForLocation)
    #print(df)
    ds = Dataset.from_pandas(df)

    tok_ds = ds.map(
        tokenize_and_align_labels, 
        batched=True,
        remove_columns=ds.column_names,
        fn_kwargs={
            "tokenizer":tokenizer, 
            "label_key": "ner_tags",
            }
    )

    print(tok_ds)

    eval_dataloader = torch.utils.data.DataLoader(
        tok_ds,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    model.eval()
    with torch.no_grad():
        true_predictions, true_labels = [], []
        for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            outputs = model(**batch)
            eval_loss = outputs.loss

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            true_predictions_batch, true_labels_batch = postprocess(predictions, labels, id2label)
            true_predictions.extend(true_predictions_batch)
            true_labels.extend(true_labels_batch)
            
        metrics = calculate_metrics(true_predictions, true_labels)
        print(metrics)
