def calculate_precision(true_pos:int, false_pos:int):
    try:
        precision = true_pos / (false_pos + true_pos)
    except ZeroDivisionError:
        precision = 0
    return precision

def calculate_accuracy(true_pos:int, true_neg:int, support:int):
    try:
        accuracy = (true_pos + true_neg) / support
    except ZeroDivisionError:
        accuracy = 0
    return accuracy

def calculate_metrics(true_pos:int, false_pos:int, true_neg:int, false_neg:int):
    support = true_pos + false_pos + true_neg + false_neg

    precision = calculate_precision(true_pos, false_pos)
    accuracy = calculate_accuracy(true_pos, true_neg, support)
    
    try:
        recall = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0
    
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    
    return precision, recall, accuracy, support, f1

def print_metrics(true_pos:int, false_pos:int, true_neg:int, false_neg:int):
    (precision, recall, accuracy, support,
        f1) = calculate_metrics(true_pos, false_pos, true_neg, false_neg)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")
    print(f"Support: {support}")
    print(f"F1: {f1}")

def calculate_recall_at(k, labels, predictions):
    recall_at = 0.0
    for label, top in zip(labels, predictions):
        if label in top[:k]:
            recall_at += 1
    if len(labels) > 0:
        recall_at /= len(labels)

    return recall_at