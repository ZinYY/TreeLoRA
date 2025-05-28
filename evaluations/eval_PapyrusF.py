import json
from metrics import caculate_f1


# resolving key words
def resolve(dataset: list):
    keyword_list = []
    for datium in dataset:
        keyword_list.append(datium.split(" , "))
    return keyword_list


def eval(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    f1 = caculate_f1(outputs, gts)
    evaluation_result = {"F1": f1}
    return evaluation_result
