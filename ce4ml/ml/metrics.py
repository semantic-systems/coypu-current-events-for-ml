from seqeval.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
                             
def calculate_metrics(pred, ref):
    f1 = float(f1_score(pred, ref)*100)
    ppv = float(precision_score(pred, ref)*100)
    sen = float(recall_score(pred, ref)*100)
    acc = float(accuracy_score(pred, ref)*100)
    return {'f1':f1, 'precision':ppv, 'recall':sen, 'accuracy':acc}


# func that turns model output and reference labels to string labels for comparing
def postprocess(predictions, labels, id2label):
    predictions = predictions.cpu().clone().numpy()
    labels = labels.cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions