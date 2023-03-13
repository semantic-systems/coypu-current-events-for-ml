from typing import List, Tuple

import lightning as pl
from torch import stack, no_grad
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForTokenClassification, get_scheduler, BatchEncoding, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

from .metrics import calculate_metrics, postprocess




class LocationExtractor(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int,
        num_batches_per_epoch: int,
        warmup_length:float, # percentage of total training steps
        label_names:List[str] = ['O', 'B-LOC', 'I-LOC'],
        learning_rate: float = 2e-5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.id2label = {i: label for i, label in enumerate(label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def forward(self, inputs):
        return self.model(**inputs)


    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        val_loss = outputs.loss
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        return {"loss": val_loss, "predictions": predictions, "labels": labels}
    
    
    
    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        val_loss = outputs.loss
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        return {"loss": val_loss, "predictions": predictions, "labels": labels}


    def validation_epoch_end(self, outputs):
        metrics, loss = self.outputs2metrics_and_loss(outputs)
        self.log("val_loss", loss, prog_bar=True)

        metrics_dict = { f"val_{key}": metrics[key]
            for key in ["precision", "recall", "f1", "accuracy"]
        }
        self.log_dict(metrics_dict, prog_bar=True)
    

    def test_epoch_end(self, outputs):
        metrics, loss = self.outputs2metrics_and_loss(outputs)
        self.log("test_loss", loss, prog_bar=True)

        metrics_dict = { f"test_{key}": metrics[key]
            for key in ["precision", "recall", "f1", "accuracy"]
        }
        self.log_dict(metrics_dict, prog_bar=True)
    

    def outputs2metrics_and_loss(self, outputs):
        loss = stack([x["loss"] for x in outputs]).mean()

        predictions_gathered = self._seq_batches2padded_seqs([x["predictions"] for x in outputs])
        labels_gathered = self._seq_batches2padded_seqs([x["labels"] for x in outputs])
        predictions, labels = postprocess(
            predictions_gathered, labels_gathered, self.id2label
        )
        metrics = calculate_metrics(predictions, labels)

        return metrics, loss


    def _seq_batches2padded_seqs(self, seq_batches):
        seqs = []
        for seq_batch in seq_batches:
            seqs.extend(list(seq_batch))
        return pad_sequence(seqs, padding_value=-100)


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup)"""

        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.hparams.learning_rate
        )

        total_estimated_batches = self.hparams.num_batches_per_epoch * 4
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=total_estimated_batches * self.hparams.warmup_length,
            num_training_steps=total_estimated_batches
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
    
    def infer(self, text:str, tokenizer) -> Tuple[List[str], List[int]]:
        batch_encoding: BatchEncoding = tokenizer(text=text, return_tensors="pt")
        input_ids = batch_encoding.input_ids
        with no_grad():
            output: TokenClassifierOutput = self.model.forward(input_ids=input_ids)
        
        tokens = batch_encoding.tokens()
        predictions = output.logits.argmax(dim=-1)

        tokens = tokens[1:-1]
        predictions = predictions[0,:].tolist()[1:-1]

        return tokens, predictions
    
    def infer_batch(self, msgs:List[str], tokenizer) -> Tuple[List[List[str]], List[List[int]]]:
        tokens_batch, predictions_batch = [],[]
        for msg in msgs:
            tokens, predictions = self.infer(msg, tokenizer)
            tokens_batch.append(tokens)
            predictions_batch.append(predictions)
        return tokens_batch, predictions_batch
    

    def infer_words(self, words:List[str], tokenizer: AutoTokenizer) -> Tuple[List[str], List[int], List[int]]:
        batch_encoding: BatchEncoding = tokenizer(
            text=words, 
            is_split_into_words=True, 
            return_tensors="pt"
        )
        input_ids = batch_encoding.input_ids
        with no_grad():
            output: TokenClassifierOutput = self.model.forward(input_ids=input_ids)
        
        tokens = batch_encoding.tokens()
        predictions = output.logits.argmax(dim=-1)

        tokens = tokens[1:-1]
        predictions = predictions[0,:].tolist()[1:-1]
        word_ids = batch_encoding.word_ids()[1:-1]

        return tokens, predictions, word_ids
    
    def infer_words_batch(self, words_batch:List[List[str]], 
            tokenizer: AutoTokenizer)-> Tuple[List[List[str]], List[List[int]], List[List[int]]]:
        tokens_batch, predictions_batch, word_ids_batch = [],[],[]
        for words in words_batch:
            tokens, predictions, word_ids = self.infer_words(words, tokenizer)
            tokens_batch.append(tokens)
            predictions_batch.append(predictions)
            word_ids_batch.append(word_ids)
        return tokens_batch, predictions_batch, word_ids_batch