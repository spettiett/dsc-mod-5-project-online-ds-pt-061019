from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def print_metrics(labels, preds):
    '''
    Purpose: Print Performance Metrics
    '''
    print(f'Precision Score: {precision_score(labels, preds)}')
    print(f'Recall Score: {recall_score(labels, preds)}')
    print(f'Accuracy Score: {accuracy_score(labels, preds)}')
    print(f'F1 Score: {f1_score(labels, preds)}')
    print(f'Misclassification Rate: {(1 - accuracy_score(labels, preds))}')