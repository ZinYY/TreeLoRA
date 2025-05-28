import json
from metrics import caculate_bleu, caculate_rouge, caculate_accuracy


# resolving answer and reasoning
def resolve(dataset: list):
    answers = []
    reasonings = []
    for dat in dataset:
        datium = str(dat).strip()
        if len(datium) < 2:
            if len(datium) == 0:
                answers.append("N/A")
                reasonings.append("N/A")
            else:
                answers.append(datium[0])
                reasonings.append("N/A")
        else:
            answers.append(datium[0])  # the first char is the answer. e.g. A, B,...
            reasonings.append(datium[2:])  # A/nBecause...
    outputs = {"answers": answers, "reasonings": reasonings}
    return outputs


def eval(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    bleu_1 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 1)
    bleu_4 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 4)
    rouge = caculate_rouge(outputs["reasonings"], gts["reasonings"])
    accuracy = caculate_accuracy(outputs["answers"], gts["answers"])

    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "accuracy": accuracy}
    return evaluation_result
