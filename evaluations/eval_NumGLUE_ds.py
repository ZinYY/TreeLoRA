import json
from metrics import caculate_accuracy


def eval(predicted_sequences, ground_truths):
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
