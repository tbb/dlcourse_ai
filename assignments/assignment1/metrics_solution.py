def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    size = prediction.size
    pred_true_indices = set((prediction == True).nonzero()[0])
    pred_false_indices = set(range(size)) - pred_true_indices
    test_true_indices = set((ground_truth == True).nonzero()[0])
    test_false_indices = set(range(size)) - test_true_indices
    
    tp = len(pred_true_indices & test_true_indices)
    fp = len(pred_true_indices & test_false_indices)
    fn = len(pred_false_indices & test_true_indices)
    tn = len(pred_false_indices & test_false_indices)

    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return (prediction == ground_truth).sum() / len(prediction)
